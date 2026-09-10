# CloudFlow SSO and Security Architecture

## Authentication & Identity Federation
CloudFlow supports Federated Identity Management via Single Sign-On (SSO) on the **Enterprise Plan**.

### Supported Identity Providers (IdP)
- **Okta**: SAML 2.0 and OIDC integration with SCIM 2.0 user provisioning.
- **Microsoft Entra ID (Azure AD)**: SAML 2.0 enterprise application integration.
- **Google Workspace**: SAML 2.0 SSO.
- **Custom IdP**: Any compliant SAML 2.0 or OpenID Connect provider.

### Configuration Steps
1. Navigate to **Settings > Security & SSO** in the CloudFlow Admin Console.
2. Download the CloudFlow SP Metadata XML (`https://auth.cloudflow.io/saml/metadata`).
3. Set the ACS (Assertion Consumer Service) URL to `https://auth.cloudflow.io/saml/acs`.
4. Input your IdP Entity ID, SSO Target URL, and upload your X.509 signing certificate.
5. Enable **Just-In-Time (JIT) Provisioning** if you want new users automatically created upon first login.

### Security Controls & Policies
- **Session Duration**: Default session TTL is 12 hours. Enterprise administrators can configure session lifetimes from 15 minutes to 24 hours.
- **Multi-Factor Authentication (MFA)**: Mandatory TOTP or FIDO2/WebAuthn hardware keys can be enforced at the organization level if SSO is bypassed by emergency fallback admins.
- **Data Encryption**: AES-256 at rest; TLS 1.3 in transit with Perfect Forward Secrecy (PFS).
