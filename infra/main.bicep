// FinShield Enterprise Infrastructure as Code (Bicep)
// Deploys complete, production-grade Azure ML architecture with zero-idle-cost compute

@description('Deployment region for all resources')
param location string = resourceGroup().location

@description('Base prefix for all resource naming conventions')
param prefix string = 'finshield'

@description('Environment tag (dev, staging, prod)')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'dev'

var uniqueSuffix = substring(uniqueString(resourceGroup().id), 0, 6)
var storageName = '${prefix}st${uniqueSuffix}'
var keyVaultName = '${prefix}-kv-${uniqueSuffix}'
var acrName = '${prefix}acr${uniqueSuffix}'
var appInsightsName = '${prefix}-ai-${uniqueSuffix}'
var logAnalyticsName = '${prefix}-log-${uniqueSuffix}'
var amlWorkspaceName = '${prefix}-aml-${environment}'

// 1. Log Analytics Workspace
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// 2. Application Insights (Telemetry & Endpoint Metrics)
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
  }
}

// 3. Azure Key Vault (Credentials & Secrets Storage)
resource keyVault 'Microsoft.KeyVault/vaults@2023-02-01' = {
  name: keyVaultName
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enableRbacAuthorization: true
    enableSoftDelete: true
  }
}

// 4. Azure Data Lake Storage Gen2 (Hierarchical Namespace enabled)
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    isHnsEnabled: true // ADLS Gen2 Hierarchical Namespace
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    accessTier: 'Hot'
  }
}

// 5. Azure Container Registry (Containerized Inference)
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    adminUserEnabled: true
  }
}

// 6. Azure Machine Learning Workspace
resource amlWorkspace 'Microsoft.MachineLearningServices/workspaces@2023-08-01-preview' = {
  name: amlWorkspaceName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: 'FinShield Credit Risk MLOps Workspace'
    description: 'Enterprise Credit Risk Scoring & Explainability Platform'
    storageAccount: storageAccount.id
    keyVault: keyVault.id
    applicationInsights: appInsights.id
    containerRegistry: containerRegistry.id
    hbiWorkspace: true // High Business Impact flag for banking data
  }
}

// 7. Auto-scaling CPU Compute Cluster (Zero-Cost when Idle)
resource amlComputeCluster 'Microsoft.MachineLearningServices/workspaces/computes@2023-08-01-preview' = {
  parent: amlWorkspace
  name: 'cpu-cluster'
  location: location
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: 'Standard_DS3_v2'
      vmPriority: 'Dedicated'
      scaleSettings: {
        minNodeCount: 0 // Scales to ZERO instances when idle ($0/hr cost)
        maxNodeCount: 4
        nodeIdleTimeBeforeScaleDown: 'PT120S' // Shuts down after 2 minutes
      }
    }
  }
}

output amlWorkspaceId string = amlWorkspace.id
output storageAccountId string = storageAccount.id
output keyVaultId string = keyVault.id
