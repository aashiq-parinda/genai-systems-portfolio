variable "aws_region" {
  type        = string
  description = "Target AWS Region"
  default     = "us-east-1"
}

variable "environment" {
  type        = string
  description = "Deployment environment (production / staging)"
  default     = "production"
}

variable "cluster_name" {
  type        = string
  description = "Enterprise GenAI EKS Cluster Name"
  default     = "enterprise-genai-mesh"
}
