"""Generate weekly markdown report"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.utils.db import connect
from src.utils.paths import REPORTS_DIR


def main():
    con = connect()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)

    # Basic coverage
    total_rows = con.execute("SELECT COUNT(*) FROM timeline_hourly").fetchone()[0]
    total_feat = con.execute("SELECT COUNT(*) FROM features_hourly").fetchone()[0]

    # Last 7 days
    last7 = con.execute("""
      SELECT *
      FROM features_hourly
      WHERE ts_hour >= ?
      ORDER BY ts_hour
    """, [week_ago]).df()

    pain_events = 0
    avg_pain = None
    if not last7.empty:
        pain_events = int(((last7["pain_max"].fillna(-1) >= 5)).sum())
        avg_pain = float(last7["pain_max"].dropna().mean()) if last7["pain_max"].notna().any() else None

    # Top coefficients if present
    has_coef = con.execute("""
      SELECT COUNT(*) FROM information_schema.tables WHERE table_name='model_coefficients'
    """).fetchone()[0] > 0

    coef_lines = []
    if has_coef:
        top = con.execute("""
          SELECT feature, coef
          FROM model_coefficients
          ORDER BY abs_coef DESC
          LIMIT 12
        """).fetchall()
        for f, c in top:
            coef_lines.append(f"- `{f}`: {c:+.4f}")

    out = []
    out.append(f"# Weekly HenryFood Report\n")
    out.append(f"- Generated (UTC): {now.isoformat()}\n")
    out.append(f"## Data coverage\n")
    out.append(f"- timeline_hourly rows: **{total_rows}**\n")
    out.append(f"- features_hourly rows: **{total_feat}**\n")

    out.append("\n## Last 7 days summary\n")
    out.append(f"- Pain events (pain_max >= 5): **{pain_events}**\n")
    out.append(f"- Average pain_max (if present): **{avg_pain if avg_pain is not None else 'n/a'}**\n")

    out.append("\n## Model signals (very early / directional)\n")
    if coef_lines:
        out.extend([line + "\n" for line in coef_lines])
    else:
        out.append("- No model coefficients yet (need symptom labels + `make weekly`).\n")

    report_path = REPORTS_DIR / "weekly.md"
    report_path.write_text("".join(out), encoding="utf-8")
    print(f"wrote report: {report_path}")


if __name__ == "__main__":
    main()
