# Google Cloud Configuration

## Project Information
- **Project ID**: wheel-strategy-202506
- **Project Number**: 673878123553
- **Billing Account**: 01C768-F853B7-29233E (My Billing Account)
- **Default Region**: us-central1

## Service Account
- **Email**: github-actions@wheel-strategy-202506.iam.gserviceaccount.com
- **Purpose**: GitHub Actions deployments via Workload Identity Federation

## Workload Identity Federation
- **Pool**: github-pool
- **Provider**: github-provider
- **Provider Resource**: projects/873567797945/locations/global/workloadIdentityPools/github-pool/providers/github-provider
- **Repository**: njrun1804/wheel-trading

## Enabled APIs
- Cloud KMS API
- Container Analysis API
- Binary Authorization API
- Secret Manager API
- Cloud Asset API
- Security Command Center API
- Cloud Logging API
- Cloud Monitoring API
- IAM API
- IAM Credentials API
- Cloud Resource Manager API
- Security Token Service API

## Security Configuration
1. **Audit Logging**: Configured for all services
2. **Service Account Roles**:
   - Cloud Build Service Account
   - Cloud Run Developer
   - Artifact Registry Writer
3. **GitHub Actions Integration**: Configured with Workload Identity Federation

## Deployment
GitHub Actions can now deploy to Google Cloud using the configured service account without storing any credentials.

## Next Steps
1. Create Artifact Registry repositories as needed
2. Configure Cloud Run services
3. Set up monitoring dashboards
4. Configure alerting policies