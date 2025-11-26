# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability within this project, please follow these steps:

### 1. Do Not Create a Public Issue

Security vulnerabilities should **not** be reported via public GitHub issues to prevent exploitation before a fix is available.

### 2. Report Privately

Please report security vulnerabilities by emailing the project maintainer directly or using GitHub's private vulnerability reporting feature (if enabled).

Include the following information in your report:
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if any)

### 3. Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution Target**: Within 30 days (depending on severity)

### 4. Severity Classifications

| Severity | Description | Response Time |
|----------|-------------|---------------|
| Critical | Remote code execution, credential theft, data breach | Immediate |
| High | Authentication bypass, privilege escalation | 24-48 hours |
| Medium | Information disclosure, denial of service | 7 days |
| Low | Minor information leak, limited impact | 30 days |

## Security Considerations

This project interacts with:
- **Ancestry.com** - User credentials stored in encrypted format
- **Browser Automation** - Selenium WebDriver sessions
- **AI APIs** - Google Gemini, DeepSeek (API keys required)
- **Local Database** - SQLite with sensitive genealogical data

### Best Practices

1. **Never commit credentials** - Use `.env` files (gitignored)
2. **Encrypt sensitive data** - Use `credentials.py` encryption
3. **Limit API key scope** - Use minimum required permissions
4. **Regular updates** - Keep dependencies updated for security patches

## Acknowledgments

We appreciate responsible disclosure and will acknowledge security researchers who report vulnerabilities (with their permission).
