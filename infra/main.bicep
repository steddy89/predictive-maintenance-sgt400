// ============================================================================
// Predictive Maintenance SGT400 - Azure Infrastructure (Bicep)
// 
// Deploys: Key Vault, Container Registry, App Service, AI/ML Workspace,
//          Application Insights, Log Analytics, Managed Identity
//
// Reference: https://learn.microsoft.com/azure/azure-resource-manager/bicep/overview
// ============================================================================

@description('Environment name (dev, staging, prod)')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'dev'

@description('Azure region for all resources')
param location string = resourceGroup().location

@description('Project name prefix')
param projectName string = 'predmaint-sgt400'

// ---- Variables ----
var suffix = '${projectName}-${environment}'
var uniqueSuffix = uniqueString(resourceGroup().id, projectName, environment)

// ---- Log Analytics Workspace ----
// Ref: https://learn.microsoft.com/azure/azure-monitor/logs/log-analytics-workspace-overview
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: 'log-${suffix}'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 90
  }
}

// ---- Application Insights ----
// Ref: https://learn.microsoft.com/azure/azure-monitor/app/app-insights-overview
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-${suffix}'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
    RetentionInDays: 90
  }
}

// ---- Managed Identity ----
// Ref: https://learn.microsoft.com/azure/active-directory/managed-identities-azure-resources/overview
resource managedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-${suffix}'
  location: location
}

// ---- Key Vault ----
// Ref: https://learn.microsoft.com/azure/key-vault/general/overview
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'kv-${uniqueSuffix}'
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 90
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
  }
}

// ---- Azure Container Registry ----
// Ref: https://learn.microsoft.com/azure/container-registry/container-registry-intro
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: 'acr${uniqueSuffix}'
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    adminUserEnabled: false
  }
}

// ---- App Service Plan ----
// Ref: https://learn.microsoft.com/azure/app-service/overview-hosting-plans
resource appServicePlan 'Microsoft.Web/serverfarms@2023-01-01' = {
  name: 'plan-${suffix}'
  location: location
  sku: {
    name: 'P1v3'
    tier: 'PremiumV3'
    capacity: 1
  }
  kind: 'linux'
  properties: {
    reserved: true
  }
}

// ---- Backend App Service ----
// Ref: https://learn.microsoft.com/azure/app-service/overview
resource backendApp 'Microsoft.Web/sites@2023-01-01' = {
  name: 'app-api-${suffix}'
  location: location
  kind: 'app,linux,container'
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentity.id}': {}
    }
  }
  properties: {
    serverFarmId: appServicePlan.id
    siteConfig: {
      linuxFxVersion: 'DOCKER|${containerRegistry.properties.loginServer}/${projectName}-api:latest'
      alwaysOn: true
      healthCheckPath: '/health'
      appSettings: [
        {
          name: 'APPINSIGHTS_INSTRUMENTATIONKEY'
          value: appInsights.properties.InstrumentationKey
        }
        {
          name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
          value: appInsights.properties.ConnectionString
        }
        {
          name: 'AZURE_KEY_VAULT_URL'
          value: keyVault.properties.vaultUri
        }
        {
          name: 'ENVIRONMENT'
          value: environment
        }
        {
          name: 'DOCKER_REGISTRY_SERVER_URL'
          value: 'https://${containerRegistry.properties.loginServer}'
        }
      ]
    }
    httpsOnly: true
  }
}

// ---- Frontend App Service ----
resource frontendApp 'Microsoft.Web/sites@2023-01-01' = {
  name: 'app-web-${suffix}'
  location: location
  kind: 'app,linux,container'
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentity.id}': {}
    }
  }
  properties: {
    serverFarmId: appServicePlan.id
    siteConfig: {
      linuxFxVersion: 'DOCKER|${containerRegistry.properties.loginServer}/${projectName}-web:latest'
      alwaysOn: true
      appSettings: [
        {
          name: 'REACT_APP_API_URL'
          value: 'https://${backendApp.properties.defaultHostName}'
        }
      ]
    }
    httpsOnly: true
  }
}

// ---- Storage Account (for ML artifacts) ----
// Ref: https://learn.microsoft.com/azure/storage/common/storage-account-overview
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: 'st${uniqueSuffix}'
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
  }
}

// ---- Azure Machine Learning Workspace (AI Foundry) ----
// Ref: https://learn.microsoft.com/azure/machine-learning/how-to-manage-workspace
resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2023-10-01' = {
  name: 'mlw-${suffix}'
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: 'Predictive Maintenance SGT400 - ${environment}'
    keyVaultId: keyVault.id
    storageAccountId: storageAccount.id
    applicationInsightsId: appInsights.id
    containerRegistryId: containerRegistry.id
  }
  sku: {
    name: 'Basic'
    tier: 'Basic'
  }
}

// ---- RBAC: Key Vault Secrets User for Managed Identity ----
resource kvRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: keyVault
  name: guid(keyVault.id, managedIdentity.id, 'Key Vault Secrets User')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')
    principalId: managedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ---- RBAC: ACR Pull for Managed Identity ----
resource acrRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: containerRegistry
  name: guid(containerRegistry.id, managedIdentity.id, 'AcrPull')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
    principalId: managedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ---- Outputs ----
output keyVaultName string = keyVault.name
output keyVaultUri string = keyVault.properties.vaultUri
output containerRegistryLoginServer string = containerRegistry.properties.loginServer
output backendAppUrl string = 'https://${backendApp.properties.defaultHostName}'
output frontendAppUrl string = 'https://${frontendApp.properties.defaultHostName}'
output mlWorkspaceName string = mlWorkspace.name
output appInsightsConnectionString string = appInsights.properties.ConnectionString
output managedIdentityClientId string = managedIdentity.properties.clientId
