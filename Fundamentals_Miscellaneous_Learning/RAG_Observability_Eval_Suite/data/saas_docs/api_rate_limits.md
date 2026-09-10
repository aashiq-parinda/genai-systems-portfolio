# CloudFlow API Rate Limits & Throttling

## Rate Limiting Architecture
CloudFlow protects upstream services and ensures fair usage through a distributed Token Bucket rate limiter managed at our edge gateway.

## Limits by Subscription Tier
- **Starter Tier**: 60 requests per minute (RPM) per API key. Maximum burst capacity of 10 requests.
- **Professional Tier**: 600 requests per minute (RPM) per API key. Maximum burst capacity of 50 requests.
- **Enterprise Tier**: 6,000 requests per minute (RPM) default, expandable up to 30,000 RPM upon request.

## Rate Limit Response Headers
Every API response includes standard rate limit headers:
- `X-RateLimit-Limit`: Maximum requests permitted per sliding window.
- `X-RateLimit-Remaining`: Number of requests remaining in current window.
- `X-RateLimit-Reset`: Unix timestamp when the current window resets.
- `Retry-After`: Number of seconds to wait before retrying (returned with HTTP 429).

## Handling HTTP 429 Too Many Requests
Clients must inspect the `Retry-After` header and implement exponential backoff with full jitter to avoid cascading thundering herd failures.
