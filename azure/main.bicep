@description('Location for all resources.')
param location string = resourceGroup().location

@description('Prefix for resource names')
param prefix string = 'creditrisk'

@description('Database administrator login name')
param dbAdminUser string = 'psqladmin'

@secure()
@description('Database administrator password')
param dbAdminPassword string

var acrName = '${prefix}acr${uniqueString(resourceGroup().id)}'
var envName = '${prefix}-env'
var psqlName = '${prefix}-psql'

// 1. Azure Container Registry
resource acr 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    // Fase 0, Capa 5: sin usuario admin del registry — el pull de imágenes
    // se hace con la identidad administrada + rol AcrPull, no credenciales
    // compartidas de larga duración.
    adminUserEnabled: false
  }
}

// Identidad administrada usada por los Container Apps para hacer pull del
// ACR y leer secretos del Key Vault, sin credenciales estáticas.
resource appIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-identity'
  location: location
}

resource acrPullRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, appIdentity.id, 'AcrPull')
  scope: acr
  properties: {
    // Rol AcrPull incorporado de Azure.
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
    principalId: appIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// 2. Log Analytics Workspace (Required for Container Apps)
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${prefix}-logs'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
  }
}

// 3. Azure Container Apps Environment
resource containerAppEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: envName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// 4. PostgreSQL Flexible Server
resource postgresServer 'Microsoft.DBforPostgreSQL/flexibleServers@2023-03-01-preview' = {
  name: psqlName
  location: location
  sku: {
    name: 'Standard_B1ms'
    tier: 'Burstable'
  }
  properties: {
    version: '14'
    administratorLogin: dbAdminUser
    administratorLoginPassword: dbAdminPassword
    storage: {
      storageSizeGB: 32
    }
  }
}

// Allow all Azure services to access Postgres (for Container Apps)
resource postgresFirewall 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2023-03-01-preview' = {
  parent: postgresServer
  name: 'AllowAllAzureServices'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

// Fase 0, Capa 5: secretos gestionados en Key Vault en vez de variables de
// entorno planas en el pipeline de CD. Este Key Vault guarda hoy la
// contraseña de Postgres; la migración de la Container App para leerla vía
// secretRef (en vez de recibirla en environmentVariables, como hace hoy
// azure-cd.yml) queda pendiente como siguiente paso, no se fuerza aquí sin
// poder probarla contra una suscripción real.
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: '${prefix}-kv'
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
  }
}

resource dbPasswordSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'db-admin-password'
  properties: {
    value: dbAdminPassword
  }
}

resource keyVaultSecretsUserRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, appIdentity.id, 'KeyVaultSecretsUser')
  scope: keyVault
  properties: {
    // Rol Key Vault Secrets User incorporado de Azure (solo lectura de secretos).
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')
    principalId: appIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Database creation
resource postgresDatabase 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2023-03-01-preview' = {
  parent: postgresServer
  name: 'credit_risk'
}

output acrLoginServer string = acr.properties.loginServer
output postgresFqdn string = postgresServer.properties.fullyQualifiedDomainName
output appIdentityId string = appIdentity.id
output appIdentityClientId string = appIdentity.properties.clientId
output keyVaultName string = keyVault.name
output keyVaultUri string = keyVault.properties.vaultUri
