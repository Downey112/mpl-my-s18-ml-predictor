"""Generate the MPL S18 Power BI project (PBIP: TMDL semantic model + PBIR report).

Usage: python scripts/build_pbip.py [OUT_DIR] [DATA_FOLDER]
Defaults: OUT_DIR = <repo>/powerbi, DATA_FOLDER = <repo>/data (absolute path baked
into the DataFolder parameter; change it in Power BI under Transform data -> Edit parameters).
Edit this script rather than the generated PBIR/TMDL files.
"""
import json
import shutil
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "powerbi"
DATA_FOLDER = sys.argv[2] if len(sys.argv) > 2 else str(REPO / "data")
NAME = "MPLS18"
SM = OUT / f"{NAME}.SemanticModel"
RP = OUT / f"{NAME}.Report"
NS = uuid.UUID("6f1c9a52-6a0e-4f43-9d8e-3a1f0c2b7d11")


def tag(key):
    return str(uuid.uuid5(NS, key))


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def wjson(path, obj):
    write(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


if OUT.exists():
    # Clear contents but keep the folder itself (Power BI Desktop may hold a lock on it)
    for child in OUT.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()

# ---------------------------------------------------------------------------
# PBIP + semantic model
# ---------------------------------------------------------------------------
wjson(OUT / f"{NAME}.pbip", {
    "$schema": "https://developer.microsoft.com/json-schemas/fabric/pbip/pbipProperties/1.0.0/schema.json",
    "version": "1.0",
    "artifacts": [{"report": {"path": f"{NAME}.Report"}}],
    "settings": {"enableAutoRecovery": True},
})
wjson(SM / "definition.pbism", {
    "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/semanticModel/definitionProperties/1.0.0/schema.json",
    "version": "4.2",
    "settings": {},
})
write(SM / "definition" / "database.tmdl", "database\n\tcompatibilityLevel: 1567\n\n")

TABLES = ["Teams", "Forecast", "Placement", "SeriesOdds", "GrandFinals", "FeatureWeights", "CVMetrics"]
write(SM / "definition" / "model.tmdl", (
    "model Model\n"
    "\tculture: en-US\n"
    "\tdefaultPowerBIDataSourceVersion: powerBI_V3\n"
    "\tsourceQueryCulture: en-US\n"
    "\tdataAccessOptions\n"
    "\t\tlegacyRedirects\n"
    "\t\treturnErrorValuesAsNull\n\n"
    "annotation __PBI_TimeIntelligenceEnabled = 0\n\n"
    f"annotation PBI_QueryOrder = {json.dumps(['DataFolder'] + TABLES)}\n\n"
    "annotation PBI_ProTooling = [\"DevMode\"]\n\n"
    + "".join(f"ref table {t}\n" for t in TABLES)
))
write(SM / "definition" / "expressions.tmdl", (
    f"expression DataFolder = \"{DATA_FOLDER}\" meta [IsParameterQuery=true, Type=\"Text\", IsParameterQueryRequired=true]\n"
    f"\tlineageTag: {tag('expr.DataFolder')}\n\n"
    "\tannotation PBI_ResultType = Text\n\n"
))
write(SM / "definition" / "relationships.tmdl", (
    f"relationship {tag('rel.forecast')}\n\tfromColumn: Forecast.Team\n\ttoColumn: Teams.team\n\n"
    f"relationship {tag('rel.placement')}\n\tfromColumn: Placement.Team\n\ttoColumn: Teams.team\n\n"
))


def indent_block(text, n):
    pad = "\t" * n
    return "\n".join(pad + line if line.strip() else "" for line in text.strip("\n").splitlines())


def table_tmdl(name, columns, m_code, measures=()):
    """columns: (name, dataType, formatString|None, extra lines list)."""
    out = [f"table {name}", f"\tlineageTag: {tag('t.' + name)}", ""]
    for mname, expr, fmt, folder in measures:
        if "\n" in expr:
            out.append(f"\tmeasure '{mname}' =")
            out.append(indent_block(expr, 3))
        else:
            out.append(f"\tmeasure '{mname}' = {expr}")
        if fmt:
            out.append(f"\t\tformatString: {fmt}")
        if folder:
            out.append(f"\t\tdisplayFolder: {folder}")
        out.append(f"\t\tlineageTag: {tag('m.' + name + '.' + mname)}")
        out.append("")
    for cname, dtype, fmt, extra in columns:
        q = f"'{cname}'" if not cname.replace("_", "").isalnum() else cname
        out.append(f"\tcolumn {q}")
        out.append(f"\t\tdataType: {dtype}")
        if fmt:
            out.append(f"\t\tformatString: {fmt}")
        for e in extra:
            out.append(f"\t\t{e}")
        out.append(f"\t\tlineageTag: {tag('c.' + name + '.' + cname)}")
        out.append("\t\tsummarizeBy: none")
        out.append(f"\t\tsourceColumn: {cname}")
        out.append("")
        if dtype == "double":
            out.insert(len(out) - 1, "\t\tannotation PBI_FormatHint = {\"isGeneralNumber\":true}")
    out.append(f"\tpartition {name} = m")
    out.append("\t\tmode: import")
    out.append("\t\tsource =")
    out.append(indent_block(m_code, 4))
    out.append("")
    out.append("\tannotation PBI_ResultType = Table")
    out.append("")
    write(SM / "definition" / "tables" / f"{name}.tmdl", "\n".join(out) + "\n")


def csv_source(file):
    return (
        "Source = Csv.Document(\n"
        f"    File.Contents(DataFolder & \"\\{file}\"),\n"
        "    [Delimiter=\",\", Encoding=65001, QuoteStyle=QuoteStyle.Csv]\n"
        "),\n"
        "Headers = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),"
    )


def m_let(steps, result):
    body = "\n".join("    " + line for line in steps.splitlines())
    return f"let\n{body}\nin\n    {result}"


# Teams ---------------------------------------------------------------------
table_tmdl(
    "Teams",
    [
        ("team", "string", None, ["isKey", "sortByColumn: seed"]),
        ("full_name", "string", None, []),
        ("seed", "int64", "0", []),
        ("Record", "string", None, []),
        ("Games", "string", None, []),
        ("Game Diff", "int64", "+0;-0;0", []),
        ("final_elo", "double", "0", []),
        ("bt_strength", "double", "0.00", []),
    ],
    m_let(
        csv_source("mpl_my_s18_team_stats.csv") + "\n"
        "Typed = Table.TransformColumnTypes(Headers, {\n"
        "    {\"team\", type text}, {\"full_name\", type text}, {\"seed\", Int64.Type},\n"
        "    {\"match_w\", Int64.Type}, {\"match_l\", Int64.Type}, {\"game_w\", Int64.Type}, {\"game_l\", Int64.Type},\n"
        "    {\"final_elo\", type number}, {\"bt_strength\", type number}\n"
        "}, \"en-US\"),\n"
        "AddRecord = Table.AddColumn(Typed, \"Record\", each Text.From([match_w]) & \"-\" & Text.From([match_l]), type text),\n"
        "AddGames = Table.AddColumn(AddRecord, \"Games\", each Text.From([game_w]) & \"-\" & Text.From([game_l]), type text),\n"
        "AddDiff = Table.AddColumn(AddGames, \"Game Diff\", each [game_w] - [game_l], Int64.Type),\n"
        "Result = Table.SelectColumns(AddDiff, {\"team\", \"full_name\", \"seed\", \"Record\", \"Games\", \"Game Diff\", \"final_elo\", \"bt_strength\"})",
        "Result",
    ),
    measures=[
        ("Title Chance Seeds 4-6", "CALCULATE ( [Champion %], REMOVEFILTERS ( Teams ), Teams[seed] IN { 4, 5, 6 } )", "0.0%", "Cards"),
    ],
)

# Forecast ------------------------------------------------------------------
FC = [
    ("Reach_UBSF_%", "Reach UB Semifinal"),
    ("Reach_UBF_%", "Reach UB Final"),
    ("Reach_LBF_%", "Reach LB Final"),
    ("M8_Qual_GrandFinal_%", "Grand Final"),
    ("Champion_%", "Champion"),
    ("RunnerUp_2nd_%", "Runner-up"),
    ("3rd_Place_%", "Third"),
    ("4th_Place_%", "Fourth"),
    ("5th_6th_Place_%", "Fifth-Sixth"),
]
table_tmdl(
    "Forecast",
    [("Team", "string", None, [])] + [(new, "double", "0.00", ["isHidden"]) for _, new in FC],
    m_let(
        csv_source("mpl_my_s18_playoffs_forecast.csv") + "\n"
        "Typed = Table.TransformColumnTypes(Headers, {{\"Team\", type text}, "
        + ", ".join(f"{{\"{old}\", type number}}" for old, _ in FC)
        + "}, \"en-US\"),\n"
        "Renamed = Table.RenameColumns(Typed, {"
        + ", ".join(f"{{\"{old}\", \"{new}\"}}" for old, new in FC)
        + "})",
        "Renamed",
    ),
    measures=[
        ("Champion %", "DIVIDE ( SUM ( Forecast[Champion] ), 100 )", "0.0%", "Base"),
        ("Grand Final %", "DIVIDE ( SUM ( Forecast[Grand Final] ), 100 )", "0.0%", "Base"),
        (
            "Card Favourite",
            "VAR _top =\n"
            "    CALCULATETABLE (\n"
            "        TOPN ( 1, ALL ( Forecast ), Forecast[Champion], DESC, Forecast[Team], ASC ),\n"
            "        REMOVEFILTERS ( Teams )\n"
            "    )\n"
            "VAR _team = MAXX ( _top, Forecast[Team] )\n"
            "VAR _p = MAXX ( _top, Forecast[Champion] ) / 100\n"
            "RETURN\n"
            "    _team & \"  \" & FORMAT ( _p, \"0.0%\" )",
            None,
            "Cards",
        ),
        (
            "Card Second M8 Slot",
            "-- Teams ranked 2 and 3 by Grand Final odds; SRG is a near-lock for the first M8 slot\n"
            "VAR _top2 =\n"
            "    CALCULATETABLE (\n"
            "        TOPN ( 2, ALL ( Forecast ), Forecast[Grand Final], DESC, Forecast[Team], ASC ),\n"
            "        REMOVEFILTERS ( Teams )\n"
            "    )\n"
            "VAR _top3 =\n"
            "    CALCULATETABLE (\n"
            "        TOPN ( 3, ALL ( Forecast ), Forecast[Grand Final], DESC, Forecast[Team], ASC ),\n"
            "        REMOVEFILTERS ( Teams )\n"
            "    )\n"
            "VAR _row2 = TOPN ( 1, _top2, Forecast[Grand Final], ASC, Forecast[Team], DESC )\n"
            "VAR _row3 = TOPN ( 1, _top3, Forecast[Grand Final], ASC, Forecast[Team], DESC )\n"
            "VAR _second = MAXX ( _row2, Forecast[Team] )\n"
            "VAR _secondVal = MAXX ( _row2, Forecast[Grand Final] )\n"
            "VAR _third = MAXX ( _row3, Forecast[Team] )\n"
            "VAR _thirdVal = MAXX ( _row3, Forecast[Grand Final] )\n"
            "RETURN\n"
            "    _second & \" \" & FORMAT ( _secondVal / 100, \"0%\" ) & \"  vs  \" & _third & \" \" & FORMAT ( _thirdVal / 100, \"0%\" )",
            None,
            "Cards",
        ),
    ],
)

# Placement (unpivoted finishing distribution) ------------------------------
table_tmdl(
    "Placement",
    [
        ("Team", "string", None, []),
        ("Finish", "string", None, ["sortByColumn: FinishOrder"]),
        ("FinishOrder", "int64", "0", ["isHidden"]),
        ("Probability", "double", "0.0%", ["isHidden"]),
    ],
    m_let(
        csv_source("mpl_my_s18_playoffs_forecast.csv") + "\n"
        "Kept = Table.SelectColumns(Headers, {\"Team\", \"Champion_%\", \"RunnerUp_2nd_%\", \"3rd_Place_%\", \"4th_Place_%\", \"5th_6th_Place_%\"}),\n"
        "Typed = Table.TransformColumnTypes(Kept, {{\"Team\", type text}, {\"Champion_%\", type number}, {\"RunnerUp_2nd_%\", type number}, "
        "{\"3rd_Place_%\", type number}, {\"4th_Place_%\", type number}, {\"5th_6th_Place_%\", type number}}, \"en-US\"),\n"
        "Unpivoted = Table.UnpivotOtherColumns(Typed, {\"Team\"}, \"Key\", \"Pct\"),\n"
        "Lookup = #table(type table [LookupKey = text, Finish = text, FinishOrder = Int64.Type], {\n"
        "    {\"Champion_%\", \"Champion\", 1}, {\"RunnerUp_2nd_%\", \"Runner-up\", 2}, {\"3rd_Place_%\", \"3rd\", 3},\n"
        "    {\"4th_Place_%\", \"4th\", 4}, {\"5th_6th_Place_%\", \"5th-6th\", 5}\n"
        "}),\n"
        "Joined = Table.Join(Unpivoted, \"Key\", Lookup, \"LookupKey\"),\n"
        "AddProb = Table.AddColumn(Joined, \"Probability\", each [Pct] / 100, type number),\n"
        "Result = Table.TransformColumnTypes(Table.SelectColumns(AddProb, {\"Team\", \"Finish\", \"FinishOrder\", \"Probability\"}), {{\"Finish\", type text}, {\"FinishOrder\", Int64.Type}})",
        "Result",
    ),
    measures=[("Placement %", "SUM ( Placement[Probability] )", "0.0%", None)],
)

# Series odds ---------------------------------------------------------------
table_tmdl(
    "SeriesOdds",
    [
        ("order", "int64", "0", []),
        ("stage", "string", None, []),
        ("Matchup", "string", None, []),
        ("best_of", "int64", "0", []),
        ("rs_h2h_matches", "string", None, []),
        ("rs_h2h_games", "string", None, []),
        ("team_a_game_win", "double", "0.0%", []),
        ("team_a_series_win", "double", "0.0%", []),
        ("Top Score", "string", None, []),
    ],
    m_let(
        csv_source("mpl_my_s18_series_odds.csv") + "\n"
        "Typed = Table.TransformColumnTypes(Headers, {{\"order\", Int64.Type}, {\"stage\", type text}, {\"team_a\", type text}, {\"team_b\", type text}, "
        "{\"best_of\", Int64.Type}, {\"rs_h2h_matches\", type text}, {\"rs_h2h_games\", type text}, {\"team_a_game_win_pct\", type number}, "
        "{\"team_a_series_win_pct\", type number}, {\"top_scoreline\", type text}, {\"top_scoreline_pct\", type number}}, \"en-US\"),\n"
        "AddMatchup = Table.AddColumn(Typed, \"Matchup\", each [team_a] & \" vs \" & [team_b], type text),\n"
        "AddGame = Table.AddColumn(AddMatchup, \"team_a_game_win\", each [team_a_game_win_pct] / 100, type number),\n"
        "AddSeries = Table.AddColumn(AddGame, \"team_a_series_win\", each [team_a_series_win_pct] / 100, type number),\n"
        "AddTop = Table.AddColumn(AddSeries, \"Top Score\", each [top_scoreline] & \" (\" & Number.ToText([top_scoreline_pct], \"0\", \"en-US\") & \"%)\", type text),\n"
        "Result = Table.SelectColumns(AddTop, {\"order\", \"stage\", \"Matchup\", \"best_of\", \"rs_h2h_matches\", \"rs_h2h_games\", \"team_a_game_win\", \"team_a_series_win\", \"Top Score\"})",
        "Result",
    ),
)

# Grand final matchups ------------------------------------------------------
table_tmdl(
    "GrandFinals",
    [
        ("rank", "int64", "0", []),
        ("Matchup", "string", None, []),
        ("share", "double", "0.0%", []),
    ],
    m_let(
        csv_source("mpl_my_s18_grand_final_matchups.csv") + "\n"
        "Typed = Table.TransformColumnTypes(Headers, {{\"rank\", Int64.Type}, {\"team_1\", type text}, {\"team_2\", type text}, {\"share_of_sims_pct\", type number}}, \"en-US\"),\n"
        "AddMatchup = Table.AddColumn(Typed, \"Matchup\", each [team_1] & \" vs \" & [team_2], type text),\n"
        "AddShare = Table.AddColumn(AddMatchup, \"share\", each [share_of_sims_pct] / 100, type number),\n"
        "Result = Table.SelectColumns(AddShare, {\"rank\", \"Matchup\", \"share\"})",
        "Result",
    ),
    measures=[
        (
            "Card Top Grand Final",
            "VAR _top = TOPN ( 1, ALL ( GrandFinals ), GrandFinals[share], DESC )\n"
            "RETURN\n"
            "    MAXX ( _top, GrandFinals[Matchup] ) & \"  \" & FORMAT ( MAXX ( _top, GrandFinals[share] ), \"0.0%\" )",
            None,
            "Cards",
        ),
    ],
)

# Feature weights -----------------------------------------------------------
table_tmdl(
    "FeatureWeights",
    [
        ("feature", "string", None, ["isHidden"]),
        ("label", "string", None, []),
        ("weight", "double", "+0.00;-0.00;0.00", ["isHidden"]),
    ],
    m_let(
        csv_source("mpl_my_s18_feature_weights.csv") + "\n"
        "Result = Table.TransformColumnTypes(Headers, {{\"feature\", type text}, {\"label\", type text}, {\"weight\", type number}}, \"en-US\")",
        "Result",
    ),
    measures=[
        ("Feature Weight", "SUM ( FeatureWeights[weight] )", "+0.00;-0.00;0.00", None),
        ("Feature Weight Color", "IF ( [Feature Weight] >= 0, \"#34D399\", \"#F87171\" )", None, None),
    ],
)

# CV metrics ----------------------------------------------------------------
table_tmdl(
    "CVMetrics",
    [
        ("model", "string", None, []),
        ("accuracy", "double", "0.000", []),
        ("roc_auc", "double", "0.000", []),
        ("brier", "double", "0.000", []),
        ("log_loss", "double", "0.000", []),
    ],
    m_let(
        csv_source("mpl_my_s18_cv_metrics.csv") + "\n"
        "Result = Table.TransformColumnTypes(Headers, {{\"model\", type text}, {\"accuracy\", type number}, {\"roc_auc\", type number}, {\"brier\", type number}, {\"log_loss\", type number}}, \"en-US\")",
        "Result",
    ),
)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
VC = "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/visualContainer/2.3.0/schema.json"
# Esports palette: deep navy canvas, dark panels with gold edges, gold/cyan highlights
DARK, MUTED, CANVAS = "#F1F5F9", "#8DA2C0", "#0A1222"
PANEL, PANEL_ALT, EDGE, GRID = "#111D35", "#0E1830", "#C9A04A", "#22324F"
GOLD, CYAN, SILVER, BRONZE = "#F2C14E", "#3BC9F5", "#C0C7D4", "#C27C48"
BLUE = GOLD


def lit(v):
    return {"expr": {"Literal": {"Value": v}}}


def s(text):
    return lit("'" + text.replace("'", "''") + "'")


def num(n):
    return lit(f"{n}D")


def b(flag):
    return lit("true" if flag else "false")


def color(hexv):
    return {"solid": {"color": s(hexv)}}


def col(entity, prop):
    return {"Column": {"Expression": {"SourceRef": {"Entity": entity}}, "Property": prop}}


def meas(entity, prop):
    return {"Measure": {"Expression": {"SourceRef": {"Entity": entity}}, "Property": prop}}


def proj(field, display=None):
    kind = "Column" if "Column" in field else "Measure"
    ent = field[kind]["Expression"]["SourceRef"]["Entity"]
    prop = field[kind]["Property"]
    p = {"field": field, "queryRef": f"{ent}.{prop}", "nativeQueryRef": prop}
    if display:
        p["displayName"] = display
    return p


def title(text, show=True):
    props = {"show": b(show)}
    if show:
        props["text"] = s(text.upper())
    return [{"properties": props}]


def visual(page, name, vtype, x, y, w, h, z, query=None, objects=None, container=None):
    v = {"visualType": vtype}
    if query:
        v["query"] = query
    if objects:
        v["objects"] = objects
    if container:
        v["visualContainerObjects"] = container
    v["drillFilterOtherVisuals"] = True
    wjson(RP / "definition" / "pages" / page / "visuals" / name / "visual.json", {
        "$schema": VC,
        "name": name,
        "position": {"x": x, "y": y, "z": z, "height": h, "width": w, "tabOrder": z},
        "visual": v,
    })


def textbox(page, name, x, y, w, h, z, paragraphs, card=False):
    paras = [
        {"textRuns": [{"value": t, "textStyle": {"fontFamily": f, "fontSize": sz, "color": c}}]}
        for t, f, sz, c in paragraphs
    ]
    visual(
        page, name, "textbox", x, y, w, h, z,
        objects={"general": [{"properties": {"paragraphs": paras}}]},
        container={
            "title": title("", show=False),
            "background": [{"properties": {"show": b(card)}}],
            "padding": [{"properties": {k: num(16 if card else 0) for k in ("top", "bottom", "left", "right")}}],
        },
    )


def accent_bar(page, name, x, y, w, z):
    """Thin gold rule under the page header."""
    visual(
        page, name, "textbox", x, y, w, 3, z,
        objects={"general": [{"properties": {"paragraphs": []}}]},
        container={
            "title": title("", show=False),
            "background": [{"properties": {"show": b(True), "color": color(GOLD), "transparency": num(0)}}],
            "border": [{"properties": {"show": b(False)}}],
            "padding": [{"properties": {k: num(0) for k in ("top", "bottom", "left", "right")}}],
        },
    )


def page_json(name, display):
    bg = [{"properties": {"color": color(CANVAS), "transparency": num(0)}}]
    wjson(RP / "definition" / "pages" / name / "page.json", {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/page/2.0.0/schema.json",
        "name": name,
        "displayName": display,
        "displayOption": "FitToPage",
        "height": 720,
        "width": 1280,
        "objects": {"background": bg, "outspace": bg},
    })


def kpi(page, name, x, label, field, z):
    visual(
        page, name, "card", x, 72, 296, 96, z,
        query={"queryState": {"Values": {"projections": [proj(field)]}}},
        objects={"labels": [{"properties": {"fontSize": num(22), "color": color(DARK)}}],
                 "categoryLabels": [{"properties": {"show": b(False)}}]},
        container={
            "title": [{"properties": {"show": b(True), "text": s(label.upper()), "fontSize": num(9),
                                      "fontColor": color(MUTED), "alignment": s("left")}}],
        },
    )


# Page 1: forecast ------------------------------------------------------------
P1 = "forecast"
page_json(P1, "Playoffs Forecast")
textbox(P1, "header", 24, 8, 1232, 56, 1000, [
    ("MPL MY S18  ·  PLAYOFFS FORECAST", "Segoe UI Black", "20pt", GOLD),
    ("50,000 SIMULATED BRACKETS  ·  OCT 28 – NOV 1, 2026  ·  TOP 2 QUALIFY FOR THE M8 WORLD CHAMPIONSHIP",
     "Segoe UI Semibold", "9pt", MUTED),
])
accent_bar(P1, "header_rule", 24, 64, 1232, 1100)
kpi(P1, "kpi_favourite", 24, "Title favourite", meas("Forecast", "Card Favourite"), 2000)
kpi(P1, "kpi_second_slot", 336, "Race for the 2nd M8 slot (reach Grand Final)", meas("Forecast", "Card Second M8 Slot"), 2100)
kpi(P1, "kpi_top_gf", 648, "Most likely Grand Final (share of sims)", meas("GrandFinals", "Card Top Grand Final"), 2200)
kpi(P1, "kpi_outsiders", 960, "Title chance for seeds 4–6 combined", meas("Teams", "Title Chance Seeds 4-6"), 2300)

visual(
    P1, "odds_by_team", "clusteredBarChart", 24, 184, 600, 256, 3000,
    query={
        "queryState": {
            "Category": {"projections": [dict(proj(col("Teams", "team"), "Team"), active=True)]},
            "Y": {"projections": [proj(meas("Forecast", "Champion %"), "Win the title"),
                                  proj(meas("Forecast", "Grand Final %"), "Reach Grand Final (M8 slot)")]},
        },
        "sortDefinition": {"sort": [{"field": meas("Forecast", "Champion %"), "direction": "Descending"}]},
    },
    objects={
        "legend": [{"properties": {"show": b(True), "position": s("Top"), "showTitle": b(False)}}],
        "dataPoint": [{"properties": {"fill": color(CYAN)}, "selector": {"metadata": "Forecast.Grand Final %"}}],
    },
    container={"title": title("Title and M8 odds by team")},
)

visual(
    P1, "finish_distribution", "hundredPercentStackedBarChart", 640, 184, 616, 256, 3100,
    query={
        "queryState": {
            "Category": {"projections": [dict(proj(col("Teams", "team"), "Team"), active=True)]},
            "Series": {"projections": [proj(col("Placement", "Finish"))]},
            "Y": {"projections": [proj(meas("Placement", "Placement %"), "Probability")]},
            "Tooltips": {"projections": [proj(meas("Forecast", "Champion %"), "Win the title")]},
        },
        # Same team order as the odds chart: most likely champion first
        "sortDefinition": {"sort": [{"field": meas("Forecast", "Champion %"), "direction": "Descending"}]},
    },
    container={"title": title("Where each team finishes")},
)

visual(
    P1, "series_odds", "tableEx", 24, 456, 600, 240, 4000,
    query={
        "queryState": {
            "Values": {"projections": [
                proj(col("SeriesOdds", "order"), "#"),
                proj(col("SeriesOdds", "stage"), "Stage"),
                proj(col("SeriesOdds", "Matchup"), "Matchup"),
                proj(col("SeriesOdds", "rs_h2h_games"), "RS games"),
                proj(col("SeriesOdds", "team_a_game_win"), "1st team wins game"),
                proj(col("SeriesOdds", "team_a_series_win"), "1st team wins series"),
                proj(col("SeriesOdds", "Top Score"), "Likeliest score"),
            ]},
        },
        "sortDefinition": {"sort": [{"field": col("SeriesOdds", "order"), "direction": "Ascending"}]},
    },
    objects={"total": [{"properties": {"totals": b(False)}}]},
    container={"title": title("Key series odds")},
)

visual(
    P1, "standings", "tableEx", 640, 456, 616, 240, 4100,
    query={
        "queryState": {
            "Values": {"projections": [
                proj(col("Teams", "seed"), "Seed"),
                proj(col("Teams", "full_name"), "Team"),
                proj(col("Teams", "Record"), "Matches"),
                proj(col("Teams", "Games"), "Games"),
                proj(col("Teams", "Game Diff"), "Diff"),
                proj(col("Teams", "final_elo"), "Elo"),
                proj(col("Teams", "bt_strength"), "Bradley-Terry"),
            ]},
        },
        "sortDefinition": {"sort": [{"field": col("Teams", "seed"), "direction": "Ascending"}]},
    },
    objects={"total": [{"properties": {"totals": b(False)}}]},
    container={"title": title("Regular season standings (top 6 made the playoffs)")},
)

# Page 2: model check -----------------------------------------------------------
P2 = "model"
page_json(P2, "Model Check")
textbox(P2, "header", 24, 8, 1232, 56, 1000, [
    ("MODEL CHECK  ·  HOW MUCH TO TRUST THE FORECAST", "Segoe UI Black", "20pt", GOLD),
    ("LEAVE-ONE-WEEK-OUT CROSS-VALIDATION AND LOGISTIC REGRESSION WEIGHTS  ·  READ WITH THE LIMITATIONS BELOW",
     "Segoe UI Semibold", "9pt", MUTED),
])
accent_bar(P2, "header_rule", 24, 64, 1232, 1100)
visual(
    P2, "feature_weights", "clusteredBarChart", 24, 80, 600, 400, 2000,
    query={
        "queryState": {
            "Category": {"projections": [dict(proj(col("FeatureWeights", "label"), "Feature"), active=True)]},
            "Y": {"projections": [proj(meas("FeatureWeights", "Feature Weight"), "Standardised weight")]},
        },
        "sortDefinition": {"sort": [{"field": meas("FeatureWeights", "Feature Weight"), "direction": "Descending"}]},
    },
    objects={
        "dataPoint": [{
            "properties": {"fill": {"solid": {"color": {"expr": meas("FeatureWeights", "Feature Weight Color")}}}},
            "selector": {"data": [{"dataViewWildcard": {"matchingOption": 1}}]},
        }],
    },
    container={"title": title("Logistic regression weights (full-data fit, standardised)")},
)
visual(
    P2, "cv_metrics", "tableEx", 640, 80, 616, 192, 2100,
    query={
        "queryState": {
            "Values": {"projections": [
                proj(col("CVMetrics", "model"), "Model"),
                proj(col("CVMetrics", "accuracy"), "Accuracy"),
                proj(col("CVMetrics", "roc_auc"), "ROC-AUC"),
                proj(col("CVMetrics", "brier"), "Brier"),
                proj(col("CVMetrics", "log_loss"), "Log loss"),
            ]},
        },
        "sortDefinition": {"sort": [{"field": col("CVMetrics", "roc_auc"), "direction": "Descending"}]},
    },
    objects={"total": [{"properties": {"totals": b(False)}}]},
    container={"title": title("Cross-validation (56 matches, 112 directed rows)")},
)
textbox(P2, "cv_note", 640, 288, 616, 192, 2200, [
    ("WHY THE ML SCORES LOOK BETTER THAN THEY ARE", "Segoe UI Bold", "11pt", GOLD),
    ("The macro features (towers, lords, kills, gold per game) are whole-season averages, so every training row already contains the result it predicts. "
     "The pre-match Elo row uses only past matches, so it is the fair bar: ROC-AUC 0.675.", "Segoe UI", "9pt", MUTED),
])
textbox(P2, "limitations", 24, 496, 1232, 200, 3000, [
    ("LIMITATIONS", "Segoe UI Bold", "11pt", GOLD),
    ("•  Elo gets a negative weight (−1.06) because season averages already encode team strength; a bigger Elo lead slightly lowers the ML probability.",
     "Segoe UI", "9pt", MUTED),
    ("•  The final game probability blends ML 45%, Bradley-Terry 30%, Elo 15% and head-to-head 10%. These weights and the 0.68 match-to-game factor are judgement calls, not fitted.",
     "Segoe UI", "9pt", MUTED),
    ("•  56 matches is a small sample. Treat the forecast as an informed opinion, not a precise estimate.",
     "Segoe UI", "9pt", MUTED),
    ("•  Data: 56 regular-season results (checked against Liquipedia) and team season totals from my.mpl.mobilelegends.com.",
     "Segoe UI", "9pt", MUTED),
], card=True)

# Report-level files ------------------------------------------------------------
wjson(RP / "definition.pbir", {
    "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definitionProperties/2.0.0/schema.json",
    "version": "4.0",
    "datasetReference": {"byPath": {"path": f"../{NAME}.SemanticModel"}},
})
wjson(RP / "definition" / "version.json", {
    "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/versionMetadata/1.0.0/schema.json",
    "version": "2.0.0",
})
wjson(RP / "definition" / "pages" / "pages.json", {
    "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/pagesMetadata/1.0.0/schema.json",
    "pageOrder": [P1, P2],
    "activePageName": P1,
})
wjson(RP / "definition" / "report.json", {
    "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/report/3.0.0/schema.json",
    "themeCollection": {"customTheme": {
        "name": f"{NAME}.json",
        "reportVersionAtImport": {"visual": "2.3.0", "report": "3.0.0", "page": "2.0.0"},
        "type": "RegisteredResources",
    }},
    "objects": {"outspacePane": [{"properties": {"expanded": b(False)}}]},
    "resourcePackages": [{"name": "RegisteredResources", "type": "RegisteredResources",
                          "items": [{"name": f"{NAME}.json", "path": f"{NAME}.json", "type": "CustomTheme"}]}],
    "settings": {"useEnhancedTooltips": True},
})

axis_muted = {"labelColor": {"solid": {"color": MUTED}}, "fontSize": 9, "fontFamily": "Segoe UI", "showAxisTitle": False}
wjson(RP / "StaticResources" / "RegisteredResources" / f"{NAME}.json", {
    "name": NAME,
    # Series order follows the finishing order: champion gold, runner-up silver, 3rd bronze
    "dataColors": [GOLD, SILVER, BRONZE, "#6B9BE8", "#7A8AA6", CYAN],
    "background": PANEL,
    "foreground": DARK,
    "backgroundLight": PANEL_ALT,
    "foregroundNeutralSecondary": MUTED,
    "tableAccent": GOLD,
    "good": "#34D399",
    "neutral": MUTED,
    "bad": "#F87171",
    "textClasses": {
        "callout": {"fontSize": 22, "fontFace": "'Segoe UI Bold', wf_segoe-ui_bold, helvetica, arial, sans-serif", "color": DARK},
        "title": {"fontSize": 11, "fontFace": "'Segoe UI Bold', wf_segoe-ui_bold, helvetica, arial, sans-serif", "color": GOLD},
        "header": {"fontSize": 10, "fontFace": "'Segoe UI Semibold', wf_segoe-ui_semibold, helvetica, arial, sans-serif", "color": GOLD},
        "label": {"fontSize": 9, "fontFace": "Segoe UI", "color": MUTED},
    },
    "visualStyles": {
        "page": {"*": {
            "background": [{"color": {"solid": {"color": CANVAS}}, "transparency": 0}],
            "outspace": [{"color": {"solid": {"color": CANVAS}}, "transparency": 0}],
        }},
        "*": {"*": {
            "background": [{"show": True, "color": {"solid": {"color": PANEL}}, "transparency": 0}],
            "border": [{"show": True, "color": {"solid": {"color": EDGE}}, "radius": 8, "width": 1}],
            "dropShadow": [{"show": False}],
            "padding": [{"top": 12, "bottom": 12, "left": 16, "right": 16}],
            "title": [{"show": True, "fontColor": {"solid": {"color": GOLD}}, "fontSize": 11,
                       "fontFamily": "'Segoe UI Bold', wf_segoe-ui_bold, helvetica, arial, sans-serif", "alignment": "left"}],
        }},
        "textbox": {"*": {
            "border": [{"show": False}],
        }},
        "card": {"*": {
            "labels": [{"color": {"solid": {"color": DARK}}, "fontSize": 22, "fontFamily": "'Segoe UI Bold', wf_segoe-ui_bold, helvetica, arial, sans-serif"}],
            "categoryLabels": [{"show": False}],
        }},
        "clusteredBarChart": {"*": {
            "categoryAxis": [dict(axis_muted, fontSize=10, gridlineShow=False, labelColor={"solid": {"color": DARK}})],
            "valueAxis": [{"show": False, "gridlineShow": False, "showAxisTitle": False}],
            "labels": [{"show": True, "color": {"solid": {"color": DARK}}, "fontSize": 9, "labelPosition": "OutsideEnd"}],
            "legend": [{"show": False, "labelColor": {"solid": {"color": MUTED}}, "fontSize": 9}],
        }},
        "hundredPercentStackedBarChart": {"*": {
            "categoryAxis": [dict(axis_muted, fontSize=10, gridlineShow=False, labelColor={"solid": {"color": DARK}})],
            "valueAxis": [{"show": False, "gridlineShow": False, "showAxisTitle": False}],
            "labels": [{"show": True, "fontSize": 8, "color": {"solid": {"color": CANVAS}}}],
            "legend": [{"show": True, "position": "Top", "showTitle": False,
                        "labelColor": {"solid": {"color": MUTED}}, "fontSize": 9}],
        }},
        "tableEx": {"*": {
            "grid": [{"gridVertical": False, "gridHorizontal": True,
                      "gridHorizontalColor": {"solid": {"color": GRID}}, "gridHorizontalWeight": 1,
                      "rowPadding": 4, "outlineColor": {"solid": {"color": GRID}}, "textSize": 9}],
            "columnHeaders": [{"fontColor": {"solid": {"color": GOLD}}, "backColor": {"solid": {"color": PANEL}},
                               "fontFamily": "'Segoe UI Bold', wf_segoe-ui_bold, helvetica, arial, sans-serif", "fontSize": 9}],
            "values": [{"fontColor": {"solid": {"color": DARK}}, "backColor": {"solid": {"color": PANEL}},
                        "fontColorSecondary": {"solid": {"color": DARK}},
                        "backColorSecondary": {"solid": {"color": PANEL_ALT}}, "fontSize": 9}],
        }},
    },
})
print("written", sum(1 for _ in OUT.rglob("*") if _.is_file()), "files to", OUT)
