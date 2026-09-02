output "cluster_endpoint" {
  description = "EKS Control Plane Kubernetes API endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "EKS Cluster identifier"
  value       = module.eks.cluster_name
}

output "vpc_id" {
  description = "Provisioned VPC ID"
  value       = module.vpc.vpc_id
}

output "private_subnets" {
  description = "Isolated private subnets housing GPU compute node pools"
  value       = module.vpc.private_subnets
}
