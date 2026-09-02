# ==============================================================================
# Enterprise GenAI Infrastructure-as-Code (Terraform)
# Provisions:
#   1. Multi-AZ VPC with Private Isolated Subnets
#   2. AWS EKS Kubernetes Cluster
#   3. Managed GPU Node Groups (SLM L4/A10G & Frontier H100/A100 with NVLink)
# ==============================================================================

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = var.environment
      Platform    = "Enterprise-GenAI-Control-Plane"
      ManagedBy   = "Terraform"
    }
  }
}

# ------------------------------------------------------------------------------
# 1. VPC & Network Architecture
# ------------------------------------------------------------------------------
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.cluster_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway   = true
  single_nat_gateway   = false  # Multi-AZ HA NAT Gateways
  enable_dns_hostnames = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
    "karpenter.sh/discovery"          = var.cluster_name
  }
}

# ------------------------------------------------------------------------------
# 2. AWS EKS Kubernetes Cluster
# ------------------------------------------------------------------------------
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.28"

  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # ----------------------------------------------------------------------------
  # EKS Managed Node Groups
  # ----------------------------------------------------------------------------
  eks_managed_node_groups = {
    # Control Plane & Ingress Gateway Tier (CPU)
    gateway_pool = {
      name           = "gateway-cpu-pool"
      instance_types = ["c6i.2xlarge"]

      min_size     = 2
      max_size     = 10
      desired_size = 3

      labels = {
        role = "control-plane-gateway"
      }
    }

    # Tier 1: SLM Serving Cluster (NVIDIA L4 / A10G for 8B FP8 Workloads)
    slm_gpu_pool = {
      name           = "slm-8b-gpu-pool"
      instance_types = ["g6.2xlarge"] # 1x NVIDIA L4 (24GB VRAM)
      ami_type       = "AL2_x86_64_GPU"

      min_size     = 2
      max_size     = 20
      desired_size = 4

      labels = {
        tier        = "slm-serving"
        accelerator = "nvidia-l4"
      }

      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }

    # Tier 2: Frontier Serving Cluster (NVIDIA H100 SXM 80GB with NVLink, TP=2)
    frontier_gpu_pool = {
      name           = "frontier-70b-gpu-pool"
      instance_types = ["p5.48xlarge"] # 8x NVIDIA H100 (640GB VRAM, 900GB/s NVLink)
      ami_type       = "AL2_x86_64_GPU"

      min_size     = 1
      max_size     = 8
      desired_size = 2

      labels = {
        tier        = "frontier-reasoning"
        accelerator = "nvidia-h100"
        tp_size     = "2"
      }

      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }
}
