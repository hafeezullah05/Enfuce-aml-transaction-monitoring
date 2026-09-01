# Part 3 — Production Architecture

A daily batch pipeline on AWS. The design goal is a **reliable, auditable,
low-operational-burden** system for a small data team — not the most scalable
possible system.

## Guiding decision: batch, not streaming

Transaction *monitoring* is post-event — the label and the workflow both come
after the transaction settles, and there is no accept/decline decision to make in
real time (ADR-0002). Batch is simpler, cheaper, reproducible, and the
entity-history features are naturally a batch computation. A streaming path only
earns its complexity if the business needs faster interdiction on a specific
high-risk corridor — at which point I'd add streaming for that subset, not
rebuild everything.

## Components

```
              ┌─────────────┐
  transactions│   S3 raw    │  partitioned by date, KMS-encrypted, immutable
     ────────►│  (landing)  │
              └──────┬──────┘
                     ▼
              ┌─────────────┐   Glue job + Great Expectations / Deequ
              │  validate   │   schema, volume vs 30-day norm, null rates
              └──────┬──────┘   fail → quarantine + page, do not proceed
                     ▼
              ┌─────────────┐   Glue (PySpark) or EMR Serverless
              │  features   │   causal entity features → S3 curated (Parquet/Iceberg)
              └──────┬──────┘
                     ▼
              ┌─────────────┐   SageMaker Batch Transform (or Processing job)
              │   score     │   loads models:/aml-transaction-monitoring/Production
              └──────┬──────┘   from the MLflow registry
                     ▼
              ┌─────────────┐   rank day's scores, apply threshold
              │   alerts    │──► DynamoDB + SQS ──► case-management tool
              └──────┬──────┘                        (SHAP reason attached)
                     ▼
              ┌─────────────┐   Evidently / SageMaker Model Monitor in a Processing job
              │  monitor    │──► CloudWatch metrics + dashboard
              └──────┬──────┘──► alarm → SNS → retraining Step Function
                     ▼
   dispositions & SARs (later) ──► S3 labels ──► used by monitor + retraining

  Orchestration:  Step Functions (daily schedule) for the pipeline above;
                  a second Step Function for retrain → shadow → approve → promote.
  Tracking/registry: MLflow server on ECS Fargate + RDS (metadata) + S3 (artifacts).
  IaC: Terraform.  Secrets: Secrets Manager.  Audit: CloudTrail + immutable S3.
```

## Key technology choices and trade-offs

| Decision | Choice | Alternative | Why this way |
|---|---|---|---|
| Serving pattern | Daily **batch** | Streaming (Kinesis + online scoring) | Post-event problem; batch is simpler and sufficient. Streaming adds an online feature store and train/serve skew risk for no benefit here. |
| Training & scoring compute | **SageMaker** (Training Job + Batch Transform) | EKS + Kubeflow | Managed, minimal ops for a small team. EKS if the org already runs Kubernetes and wants portability/cost control at scale. |
| Registry & tracking | **MLflow** self-hosted | SageMaker Model Registry | One tool for tracking *and* registry, portable across clouds, and the standard the team uses. Cost: we run the server (Fargate + RDS). SM Registry is less to operate but ties us to SageMaker. |
| Feature compute | **Glue / EMR Serverless** (PySpark) | Pandas on a big instance; dbt/Snowpark | Spark scales with volume and is serverless-managed. Pandas is fine at today's 9.5M rows but not at 10x. dbt/Snowpark if the data already lives in Snowflake. |
| Feature storage | S3 **Iceberg/Parquet** table, code shared train↔serve | SageMaker Feature Store | Batch-only, no online serving — an offline table plus a shared feature module gives parity without the infrastructure. |
| Orchestration | **Step Functions** | Airflow / MWAA | Serverless, native retries and alarms, cheaper to run. Airflow if the team wants a richer DAG ecosystem and already operates it. |
| Case management | **Buy** (Actimize / existing tool) | Build a UI | Not where the differentiation is; investigators want it in their existing workflow. |
| Data validation | **Great Expectations / Deequ** | ad-hoc checks | A broken feed is the most common cause of "model failure"; explicit contracts catch it before scoring. |

## Cross-cutting

- **Security & PII**: KMS on every bucket, VPC-isolated compute, IAM least
  privilege, account numbers tokenised in the curated zone, CloudTrail for audit.
- **Cost**: everything serverless/scheduled — no idle clusters. Dominant cost is
  the daily feature + scoring job (~minutes of Spark/SageMaker) and the MLflow
  RDS instance.
- **Reliability**: each Step Functions state has retries + a dead-letter path; a
  failed validation blocks scoring rather than scoring bad data; runbooks for
  feed failure, drift alarm, challenger regression.
- **Reproducibility**: raw zone is immutable and date-partitioned, so any past
  batch can be re-scored with any model version.

## Minimal Terraform sketch

```hcl
# Zones
resource "aws_s3_bucket" "raw"     { bucket = "enfuce-aml-raw" }
resource "aws_s3_bucket" "curated" { bucket = "enfuce-aml-curated" }
resource "aws_s3_bucket" "artifacts" { bucket = "enfuce-aml-mlflow-artifacts" }

resource "aws_s3_bucket_server_side_encryption_configuration" "raw" {
  bucket = aws_s3_bucket.raw.id
  rule { apply_server_side_encryption_by_default { sse_algorithm = "aws:kms" } }
}

# Daily pipeline
resource "aws_sfn_state_machine" "daily" {
  name     = "aml-daily"
  role_arn = aws_iam_role.sfn.arn
  definition = jsonencode({
    StartAt = "Validate"
    States = {
      Validate = { Type = "Task", Resource = aws_glue_job.validate.arn, Next = "Features" }
      Features = { Type = "Task", Resource = aws_glue_job.features.arn, Next = "Score" }
      Score    = { Type = "Task", Resource = "arn:aws:states:::sagemaker:createTransformJob.sync", Next = "Alert" }
      Alert    = { Type = "Task", Resource = aws_lambda_function.rank_and_alert.arn, Next = "Monitor" }
      Monitor  = { Type = "Task", Resource = aws_glue_job.monitor.arn, End = true }
    }
  })
}

resource "aws_scheduler_schedule" "daily" {
  schedule_expression = "cron(0 6 * * ? *)"
  target { arn = aws_sfn_state_machine.daily.arn, role_arn = aws_iam_role.scheduler.arn }
}

resource "aws_cloudwatch_metric_alarm" "alert_rate" {
  alarm_name          = "aml-alert-rate-out-of-band"
  namespace           = "AML/Monitor"
  metric_name         = "alert_rate"
  threshold           = 0.015
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  alarm_actions       = [aws_sns_topic.retrain.arn]
}
```
