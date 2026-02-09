# Bundled Fallback Data

Fallback CSVs used when live API sources (FRED, World Bank) are unreachable.

## Sources
- **cpi.csv** – Pakistan annual CPI inflation (% YoY). Source: World Bank API, indicator `FP.CPI.TOTL.ZG`. Retrieved Feb 2026.
- **tbill.csv** – SBP policy rate annual averages (decimal). Source: State Bank of Pakistan monetary policy decisions, cross-referenced with Trading Economics and CEIC Data. Retrieved Feb 2026.

## Notes
- The pipeline tries live APIs first; these files are only used if those fail.
- Update periodically to include newer years as data becomes available.
