## Deployment Instructions

This project uses GitHub Actions to deploy a Dockerized Flask application to AWS Elastic Beanstalk. Follow these steps to deploy it on your AWS account:

### Prerequisites
- A GitHub account and repository (fork or clone this repo).
- An AWS account with access to Elastic Beanstalk and S3.
- Docker installed locally (optional, for testing).

### Steps
1. **Set Up AWS Resources**
   - **S3 Bucket:**
     - Create an S3 bucket named `food-delivery-deploy` in your AWS account.
   - **Elastic Beanstalk:**
     - Go to the Elastic Beanstalk console.
     - Create an application named `food-delivery-prediction`.
     - Create an environment named `food-delivery-prod` with the "Docker" platform (use a sample app initially).
   - **IAM User:**
     - Create an IAM user with `AWSElasticBeanstalkFullAccess` and `AmazonS3FullAccess` permissions.
     - Note the Access Key ID and Secret Access Key.

2. **Configure GitHub Secrets**
   - In your GitHub repository, go to **Settings > Secrets and variables > Actions > Secrets**.
   - Add the following secrets:
     - `AWS_ACCESS_KEY_ID`: Your IAM user's Access Key ID.
     - `AWS_SECRET_ACCESS_KEY`: Your IAM user's Secret Access Key.

3. **Push Changes**
   - Ensure the `Dockerfile` and `.github/workflows/deploy.yml` are in your repository.
   - Push changes to the `main` branch:
     ```bash
     git add .
     git commit -m "Set up CI/CD with GitHub Actions"
     git push origin main