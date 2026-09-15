# Security policy

## Reporting a vulnerability

Use the repository's [private vulnerability reporting form](https://github.com/huangzhilin-hzl/entelechy/security/advisories/new)
when it is available. Do not disclose an exploit, credentials, private model data,
or sensitive logs in a public issue or pull request.

If private reporting is unavailable, open an issue titled **Request for a private
security reporting channel**. Include only a request for a private contact method;
wait for that channel before sharing vulnerability details. This repository does
not publish a separate security email address.

A private report should include the affected revision, a minimal reproducer,
expected and observed behavior, impact, and relevant environment details.
Maintainers will coordinate investigation, a fix, and disclosure through the
private channel. Response times depend on maintainer availability.

## Supported development line

Security fixes target the current `main` branch. Older development snapshots do
not have a separate maintenance commitment. Update to a revision containing the
fix before continuing an affected experiment.

## Execution model

Entelechy runs native CUDA code and explicitly configured external proposer
commands. Use trusted source code and dependencies on a machine you control.
Worker subprocesses, timeouts, GPU leases, and content hashes provide operational
isolation and experiment integrity; they are not a sandbox for hostile code.

Only training inputs are sent through the proposer protocol. Repository files and
held-out data remain accessible to processes with the same filesystem permissions.
Use external process, account, or container isolation when stronger separation is
required.
