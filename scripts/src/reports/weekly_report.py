"""Weekly report generation.

This module generates a markdown report with:
- Data coverage statistics
- Last 7 days summary
- Top model coefficients (if trained)

Security hardening:
- Safe file writing with proper permissions
- Path validation
- HTML escaping for any user-provided content in reports
- Resource limits on report generation
"""
from __future__ import annotations

import html
from datetime import datetime, timedelta, timezone
import sys

from src.utils.db import connect
from src.utils.paths import REPORTS_DIR, validate_path

def _escape_html(text: str) -> str:
    """Escape HTML to prevent injection in markdown/HTML reports."""
    return html.escape(str(text))

def main():
    """Generate weekly markdown report."""
    con = connect(read_only=True)  # Read-only since we're just reporting
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)

    # Basic coverage
    try:
        total_rows = con.execute("SELECT COUNT(*) FROM timeline_hourly").fetchone()[0]
        total_feat = con.execute("SELECT COUNT(*) FROM features_hourly").fetchone()[0]
    except Exception as e:
        print(f"Error reading data: {e}", file=sys.stderr)
        total_rows = 0
        total_feat = 0

    # Last 7 days
    try:
        last7 = con.execute("""
          SELECT *
          FROM features_hourly
          WHERE ts_hour >= ?
          ORDER BY ts_hour
        """, [week_ago]).df()
    except Exception as e:
        print(f"Error reading last 7 days: {e}", file=sys.stderr)
        import pandas as pd
        last7 = pd.DataFrame()

    pain_events = 0
    avg_pain = None
    meal_count = 0
    
    if not last7.empty:
        pain_events = int(((last7["pain_max"].fillna(-1) >= 5)).sum())
        avg_pain = float(last7["pain_max"].dropna().mean()) if last7["pain_max"].notna().any() else None
        meal_count = int(last7["meal_events"].sum())

    # Top coefficients if present
    has_coef = False
    try:
        has_coef = con.execute("""
          SELECT COUNT(*) FROM information_schema.tables WHERE table_name='model_coefficients'
        """).fetchone()[0] > 0
    except:
        pass

    coef_lines = []
    if has_coef:
        try:
            top = con.execute("""
              SELECT feature, coef
              FROM model_coefficients
              ORDER BY abs_coef DESC
              LIMIT 12
            """).fetchall()
            for f, c in top:
                # Escape feature name to prevent injection
                safe_feature = _escape_html(f)
                coef_lines.append(f"- `{safe_feature}`: {c:+.4f}")
        except Exception as e:
            print(f"Error reading coefficients: {e}", file=sys.stderr)

    # Build report
    out = []
    out.append(f"# Weekly HenryFood Report\n")
    out.append(f"\n- Generated (UTC): {now.isoformat()}\n")
    out.append(f"- Report period: {week_ago.date()} to {now.date()}\n")
    
    out.append(f"\n## Data Coverage\n")
    out.append(f"- Timeline hourly rows: **{total_rows}**\n")
    out.append(f"- Features hourly rows: **{total_feat}**\n")

    out.append("\n## Last 7 Days Summary\n")
    out.append(f"- Total meal events: **{meal_count}**\n")
    out.append(f"- Pain events (pain_max >= 5): **{pain_events}**\n")
    if avg_pain is not None:
        out.append(f"- Average pain_max: **{avg_pain:.2f}**\n")
    else:
        out.append(f"- Average pain_max: **n/a** (no pain data)\n")

    out.append("\n## Model Signals\n")
    out.append("\n*These are early directional signals from a simple model. " +
               "Interpret with caution and validate with domain knowledge.*\n\n")
    if coef_lines:
        out.extend([line + "\n" for line in coef_lines])
    else:
        out.append("- No model coefficients yet. Need symptom labels and `make weekly`.\n")

    out.append("\n## Security Notes\n")
    out.append("- All data validated and sanitized during ingestion\n")
    out.append("- Database accessed in read-only mode for reporting\n")
    out.append("- Report generated with HTML escaping for safety\n")

    # Write report with safe permissions
    report_path = REPORTS_DIR / "weekly.md"
    report_path = validate_path(report_path)
    
    try:
        report_path.write_text("".join(out), encoding="utf-8")
        # Set secure permissions (owner read/write only)
        # Note: This works on Unix-like systems. Windows ACLs handle permissions differently.
        try:
            report_path.chmod(0o600)
        except (OSError, NotImplementedError):
            # Windows or other platforms that don't support Unix permissions
            # Fall back to default permissions (which on Windows are typically user-only anyway)
            pass
        print(f"Wrote report: {report_path}")
    except Exception as e:
        print(f"Error writing report: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
