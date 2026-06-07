"""
build_report_html.py — Render reports/report_data.json into a standalone HTML report.

The JSON is embedded directly into the HTML so the file is fully portable
(open it with file:// — no server, no fetch, works offline).

The report is bilingual (English / Spanish) with a live toggle; all display
strings and team names are translated client-side, so the underlying data
(English team keys) stays canonical.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DATA = json.load(open(os.path.join(HERE, "report_data.json"), encoding="utf-8"))
# Standalone deliverable + the copy the web app serves statically from frontend/.
OUT  = os.path.join(HERE, "wc_2026_consensus_report.html")
WEB_OUT = os.path.join(ROOT, "frontend", "report.html")

DATA_JSON = json.dumps(DATA, ensure_ascii=False)

HTML = r"""<!DOCTYPE html>
<html lang="en" class="theme-yellow">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The AI Consensus of the World Cup — 2026 Forecast</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Gabarito:wght@400..700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg-base:#0A192F; --bg-surface:#112240; --bg-surface-2:#1E3A5F; --bg-overlay:#0D1F3C;
  --border-subtle:rgba(255,255,255,.06); --border-default:rgba(255,255,255,.10); --border-strong:rgba(255,255,255,.20);
  --text-primary:#fff; --text-secondary:rgba(255,255,255,.65); --text-muted:rgba(255,255,255,.35); --text-inverse:#1A1500;
  --color-success:#22D3A0; --color-warning:#FACC15; --color-error:#F87171; --color-info:#60A5FA;
  --accent:#FFE600; --accent-dim:rgba(255,230,0,.14); --accent-glow:rgba(255,230,0,.30); --bg-hover:rgba(255,230,0,.06);
  --font-sans:"Be Vietnam Pro",system-ui,-apple-system,sans-serif;
  --font-serif:"Gabarito","Gabarito Variable",ui-serif,serif;
  --font-mono:"JetBrains Mono",ui-monospace,monospace;
  --radius-md:8px; --radius-lg:12px; --radius-xl:18px;
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:var(--bg-base);color:var(--text-primary);font-family:var(--font-sans);
  line-height:1.55;font-weight:400;-webkit-font-smoothing:antialiased;
  background-image:radial-gradient(1200px 600px at 80% -10%,rgba(255,230,0,.06),transparent 60%);}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px}
h1,h2,h3,h4{font-family:var(--font-serif);font-weight:700;line-height:1.15;letter-spacing:-.02em}
a{color:var(--accent);text-decoration:none}
.muted{color:var(--text-secondary)} .dim{color:var(--text-muted)}
.mono{font-family:var(--font-mono)}

/* language toggle */
.lang-toggle{position:fixed;top:14px;right:16px;z-index:50;display:flex;border:1px solid var(--border-strong);
  border-radius:999px;overflow:hidden;background:rgba(10,25,47,.8);backdrop-filter:blur(8px)}
.lang-toggle button{background:none;border:none;color:var(--text-secondary);font-family:var(--font-mono);
  font-size:.72rem;padding:6px 13px;cursor:pointer;letter-spacing:.06em}
.lang-toggle button.active{background:var(--accent);color:var(--text-inverse);font-weight:700}

/* hero */
header.hero{padding:72px 0 40px;border-bottom:1px solid var(--border-subtle)}
.kicker{font-family:var(--font-mono);font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;color:var(--accent);margin-bottom:16px}
.hero h1{font-size:clamp(2.4rem,6vw,4rem);margin-bottom:14px}
.hero h1 .hl{color:var(--accent)}
.hero p.lead{font-size:1.12rem;color:var(--text-secondary);max-width:680px}
.hero .meta{margin-top:24px;display:flex;flex-wrap:wrap;gap:10px}
.chip{font-family:var(--font-mono);font-size:.72rem;background:var(--bg-surface);border:1px solid var(--border-default);
  padding:6px 12px;border-radius:999px;color:var(--text-secondary)}
.chip b{color:var(--text-primary)}

/* nav */
nav.toc{position:sticky;top:0;z-index:20;background:rgba(10,25,47,.86);backdrop-filter:blur(10px);
  border-bottom:1px solid var(--border-subtle);padding:12px 0;margin-bottom:8px}
nav.toc .wrap{display:flex;gap:6px;flex-wrap:wrap;padding-right:90px}
nav.toc a{font-size:.8rem;color:var(--text-secondary);padding:6px 11px;border-radius:999px;border:1px solid transparent}
nav.toc a:hover{color:var(--text-primary);background:var(--bg-hover);border-color:var(--border-default)}

section{padding:54px 0;border-bottom:1px solid var(--border-subtle);scroll-margin-top:64px}
section h2{font-size:2rem;margin-bottom:6px}
section .sub{color:var(--text-secondary);margin-bottom:28px;max-width:760px}
.eyebrow{font-family:var(--font-mono);font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);margin-bottom:10px}

.card{background:var(--bg-surface);border:1px solid var(--border-default);border-radius:var(--radius-lg);padding:20px}
.grid{display:grid;gap:16px}
.g2{grid-template-columns:repeat(2,1fr)} .g3{grid-template-columns:repeat(3,1fr)} .g4{grid-template-columns:repeat(4,1fr)}
@media(max-width:880px){.g2,.g3,.g4{grid-template-columns:1fr}}

/* bars */
.bar-row{display:grid;grid-template-columns:150px 1fr 56px;align-items:center;gap:12px;padding:5px 0}
.bar-row .name{font-size:.9rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bar-track{background:var(--bg-surface-2);border-radius:999px;height:18px;overflow:hidden;position:relative}
.bar-fill{height:100%;background:linear-gradient(90deg,var(--accent),#FFB800);border-radius:999px}
.bar-row .val{font-family:var(--font-mono);font-size:.82rem;text-align:right;color:var(--text-secondary)}

/* tables */
table{width:100%;border-collapse:collapse;font-size:.85rem}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--border-subtle)}
th{font-family:var(--font-mono);font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;color:var(--text-muted);font-weight:600}
td.num,th.num{text-align:right;font-family:var(--font-mono)}
tbody tr:hover{background:var(--bg-hover)}
.heat{border-radius:5px;text-align:center;font-family:var(--font-mono);font-size:.78rem;padding:6px 4px}

/* group cards */
.group-card h4{font-size:1rem;margin-bottom:2px}
.group-card .gl{font-family:var(--font-mono);color:var(--accent);font-size:.75rem;letter-spacing:.1em}
.team-line{display:grid;grid-template-columns:1fr 1fr;gap:8px;align-items:center;padding:7px 0;border-top:1px solid var(--border-subtle)}
.team-line:first-of-type{border-top:none}
.team-line .tn{font-size:.86rem}
.mini-track{background:var(--bg-surface-2);height:7px;border-radius:999px;overflow:hidden}
.mini-fill{height:100%;background:var(--accent)}
.team-line .pct{font-family:var(--font-mono);font-size:.72rem;color:var(--text-secondary);min-width:42px;text-align:right}

/* bracket */
.bracket{display:flex;gap:18px;overflow-x:auto;padding-bottom:12px}
.round{display:flex;flex-direction:column;justify-content:space-around;min-width:190px;gap:8px}
.round h4{font-size:.75rem;font-family:var(--font-mono);letter-spacing:.1em;text-transform:uppercase;color:var(--text-muted);text-align:center;margin-bottom:4px}
.match{background:var(--bg-surface);border:1px solid var(--border-default);border-radius:8px;padding:7px 9px;font-size:.8rem}
.match .t{display:flex;justify-content:space-between;gap:8px;padding:2px 0}
.match .t.win{color:var(--accent);font-weight:600}
.match .t .p{font-family:var(--font-mono);font-size:.7rem;color:var(--text-muted)}
.champ-box{background:var(--accent);color:var(--text-inverse);border-radius:10px;padding:14px;text-align:center;font-weight:700;font-family:var(--font-serif);font-size:1.2rem}

/* model bias */
.bias-cols{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:12px}
@media(max-width:720px){.bias-cols{grid-template-columns:1fr}}
.delta-row{display:grid;grid-template-columns:120px 1fr 52px;align-items:center;gap:10px;padding:4px 0;font-size:.84rem}
.delta-track{position:relative;height:14px;background:var(--bg-surface-2);border-radius:4px}
.delta-track .zero{position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--border-strong)}
.delta-pos,.delta-neg{position:absolute;top:0;height:100%;border-radius:3px}
.delta-pos{left:50%;background:var(--color-success)}
.delta-neg{right:50%;background:var(--color-error)}
.delta-row .dv{font-family:var(--font-mono);font-size:.74rem;text-align:right}

/* explorer */
.explorer-controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:20px}
select{background:var(--bg-surface);color:var(--text-primary);border:1px solid var(--border-strong);
  border-radius:8px;padding:10px 14px;font-family:var(--font-sans);font-size:1rem;min-width:240px}
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}
@media(max-width:720px){.stat-grid{grid-template-columns:repeat(2,1fr)}}
.stat{background:var(--bg-surface);border:1px solid var(--border-default);border-radius:10px;padding:14px}
.stat .v{font-family:var(--font-serif);font-size:1.7rem;font-weight:700;color:var(--accent)}
.stat .l{font-size:.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em;margin-top:2px}
.journey{background:var(--bg-overlay);border:1px solid var(--border-subtle);border-radius:8px;padding:10px 12px;margin:6px 0;
  display:flex;justify-content:space-between;gap:12px;font-size:.86rem}
.journey .jp{font-family:var(--font-mono);color:var(--accent);font-size:.78rem}
.pill{display:inline-block;font-family:var(--font-mono);font-size:.72rem;background:var(--accent-dim);color:var(--accent);
  border:1px solid var(--accent-glow);padding:3px 9px;border-radius:999px}
.note{font-size:.85rem;color:var(--text-secondary);background:var(--bg-overlay);border-left:3px solid var(--accent);
  padding:12px 16px;border-radius:0 8px 8px 0;margin-top:14px}
footer{padding:40px 0 60px;color:var(--text-muted);font-size:.82rem}
.kv{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--border-subtle);font-size:.85rem}
.kv .k{color:var(--text-secondary)} .kv .v{font-family:var(--font-mono)}
</style>
</head>
<body>
<div class="lang-toggle" id="langToggle">
  <button data-lang="en">EN</button>
  <button data-lang="es">ES</button>
</div>

<header class="hero"><div class="wrap">
  <div class="kicker" id="kicker"></div>
  <h1 id="h1"></h1>
  <p class="lead" id="lead"></p>
  <div class="meta" id="heroMeta"></div>
</div></header>

<nav class="toc"><div class="wrap">
  <a href="#odds" id="nav_odds"></a>
  <a href="#stages" id="nav_stages"></a>
  <a href="#groups" id="nav_groups"></a>
  <a href="#bracket" id="nav_bracket"></a>
  <a href="#bias" id="nav_bias"></a>
  <a href="#confidence" id="nav_conf"></a>
  <a href="#findings" id="nav_find"></a>
  <a href="#explorer" id="nav_expl"></a>
</div></nav>

<section id="odds"><div class="wrap">
  <div class="eyebrow" id="e_odds"></div>
  <h2 id="t_odds"></h2>
  <p class="sub" id="s_odds"></p>
  <div id="oddsBars"></div>
</div></section>

<section id="stages"><div class="wrap">
  <div class="eyebrow" id="e_stages"></div>
  <h2 id="t_stages"></h2>
  <p class="sub" id="s_stages"></p>
  <div style="overflow-x:auto"><table id="stageTable"></table></div>
</div></section>

<section id="groups"><div class="wrap">
  <div class="eyebrow" id="e_groups"></div>
  <h2 id="t_groups"></h2>
  <p class="sub" id="s_groups"></p>
  <div class="grid g3" id="groupGrid"></div>
</div></section>

<section id="bracket"><div class="wrap">
  <div class="eyebrow" id="e_bracket"></div>
  <h2 id="t_bracket"></h2>
  <p class="sub" id="s_bracket"></p>
  <div class="bracket" id="bracketBox"></div>
  <div class="note" id="chalkNote"></div>
</div></section>

<section id="bias"><div class="wrap">
  <div class="eyebrow" id="e_bias"></div>
  <h2 id="t_bias"></h2>
  <p class="sub" id="s_bias"></p>
  <div id="biasBox"></div>
</div></section>

<section id="confidence"><div class="wrap">
  <div class="eyebrow" id="e_conf"></div>
  <h2 id="t_conf"></h2>
  <p class="sub" id="s_conf"></p>
  <div style="overflow-x:auto"><table id="confTable"></table></div>
  <div class="note" id="confNote"></div>
</div></section>

<section id="findings"><div class="wrap">
  <div class="eyebrow" id="e_find"></div>
  <h2 id="t_find"></h2>
  <div class="grid g2" id="findGrid" style="margin:6px 0 16px"></div>
  <div class="grid g2">
    <div class="card"><h4 style="margin-bottom:10px" id="find_dis_h"></h4>
      <p class="dim" style="font-size:.8rem;margin-bottom:10px" id="find_dis_p"></p>
      <table id="disagreeTable"></table></div>
    <div class="card"><h4 style="margin-bottom:10px" id="find_coin_h"></h4>
      <p class="dim" style="font-size:.8rem;margin-bottom:10px" id="find_coin_p"></p>
      <table id="coinTable"></table></div>
  </div>
  <div class="grid g2" style="margin-top:16px">
    <div class="card"><h4 style="margin-bottom:10px" id="find_finals_h"></h4>
      <table id="finalsTable"></table></div>
    <div class="card"><h4 style="margin-bottom:10px" id="find_lop_h"></h4>
      <table id="lopTable"></table></div>
  </div>
</div></section>

<section id="explorer"><div class="wrap">
  <div class="eyebrow" id="e_expl"></div>
  <h2 id="t_expl"></h2>
  <p class="sub" id="s_expl"></p>
  <div class="explorer-controls">
    <select id="teamSelect"></select>
    <span class="pill" id="teamGroupPill"></span>
  </div>
  <div id="teamPanel"></div>
</div></section>

<footer><div class="wrap">
  <p id="foot1"></p>
  <p style="margin-top:8px" id="foot2"></p>
</div></footer>

<script id="reportData" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('reportData').textContent);
const pct = x => (x*100).toFixed(1)+'%';
const pct0 = x => Math.round(x*100)+'%';
const el = (t,c,h)=>{const e=document.createElement(t);if(c)e.className=c;if(h!=null)e.innerHTML=h;return e;};
const heat = v => `background:rgba(255,230,0,${(0.05+v*0.85).toFixed(3)});color:${v>0.55?'#1A1500':'#fff'}`;
const nfmt = n => n.toLocaleString(LANG==='es'?'es-ES':'en-US');

// ── Spanish team-name map (English data keys → display names) ────────────────
const TEAMS_ES = {
  "Algeria":"Argelia","Argentina":"Argentina","Australia":"Australia","Austria":"Austria",
  "Belgium":"Bélgica","Bosnia and Herzegovina":"Bosnia y Herzegovina","Brazil":"Brasil",
  "Canada":"Canadá","Cape Verde":"Cabo Verde","Colombia":"Colombia","Croatia":"Croacia",
  "Curaçao":"Curazao","Czechia":"Chequia","Côte d'Ivoire":"Costa de Marfil","DR Congo":"RD Congo",
  "Ecuador":"Ecuador","Egypt":"Egipto","England":"Inglaterra","France":"Francia","Germany":"Alemania",
  "Ghana":"Ghana","Haiti":"Haití","Iran":"Irán","Iraq":"Irak","Japan":"Japón","Jordan":"Jordania",
  "Mexico":"México","Morocco":"Marruecos","Netherlands":"Países Bajos","New Zealand":"Nueva Zelanda",
  "Norway":"Noruega","Panama":"Panamá","Paraguay":"Paraguay","Portugal":"Portugal","Qatar":"Catar",
  "Saudi Arabia":"Arabia Saudí","Scotland":"Escocia","Senegal":"Senegal","South Africa":"Sudáfrica",
  "South Korea":"Corea del Sur","Spain":"España","Sweden":"Suecia","Switzerland":"Suiza",
  "Tunisia":"Túnez","Turkey":"Turquía","USA":"EE. UU.","Uruguay":"Uruguay","Uzbekistan":"Uzbekistán"
};
const tn = t => (LANG==='es' && TEAMS_ES[t]) ? TEAMS_ES[t] : t;
const placeWord = r => (LANG==='es'
  ? ({1:'1.º',2:'2.º',3:'3.º',4:'4.º'})[r]
  : ({1:'1st',2:'2nd',3:'3rd',4:'4th'})[r]) || String(r);

// ── Translation tables ──────────────────────────────────────────────────────
const I18N = {
 en:{
  title:"The AI Consensus of the World Cup — 2026 Forecast",
  REACH:{ko:'Reach R32',r16:'Reach R16',qf:'Reach QF',sf:'Reach SF',final:'Reach Final',champion:'Win title'},
  roundLabels:{r32:'R32',r16:'R16',qf:'QF',sf:'SF',final:'Final'},
  exitRound:{ko:'the Round of 32',r16:'the Round of 16',qf:'the Quarter-final',sf:'the Semi-final',final:'the Final'},
  championWord:'Champion',
  jChampion:'Won the tournament',
  jGroup:p=>`${p} in group → eliminated in group stage`,
  jKO:(p,r,e)=>`${p} in group → out in ${r} (beaten by ${e})`,
  favourite:'Favourite', chalkChampion:'Chalk champion', simulationsWord:'simulations',
  modelsWord:'models', seedWord:'seed', vs:'vs',
  th_team:'Team', th_wingroup:'Win group', groupWord:'GROUP',
  groupsFoot:(teamHtml,p)=>`bar = advance · top: <b style="color:var(--accent)">${teamHtml}</b> wins group ${p}`,
  topFavs:'Top favourites:', inflates:'Inflates ▲', suppresses:'Suppresses ▼',
  confHeaders:['Model','Decisiveness','Avg top outcome','Avg draw%','|win₁−win₂|'],
  findHeaders:{matchup:'Matchup',spread:'Spread',range:'Range',win1:'Win₁',draw:'Draw',win2:'Win₂',pairing:'Pairing',prob:'Probability',fav:'Favourite'},
  stat:{winCup:'Win the cup',reachKO:'Reach knockouts',winGroup:'Win the group',topElim:'Top eliminator'},
  expl:{reachTitle:'Chance of reaching each round',
    reachSub:'"Reach" = plays in that round. "Win title" = lifts the trophy.',
    groupFinish:'Group finish', howOut:'How they go out',
    commonJourneys:'Most common journeys',
    journeysSub:n=>`The tournament storylines that recur most often across ${n} simulations.`,
    mostRepeated:'Most repeated single outcome:'},
  rankLabels:['Win group','2nd','3rd','4th'],
  elimGroupStage:'Eliminated in group stage', knockedOutBy:'Knocked out by',
  chalkNote:()=>`In the chalk bracket the favourite always advances, so it crowns <b>${tn(D.chalk.champion)}</b>. But across ${nfmt(D.meta.n_sims)} random simulations the most frequent champion is <b>${tn(D.most_common_champion)}</b> with only ${pct(D.title_odds[0].prob)} — proof that "most likely on paper" still loses far more often than it wins.`,
  confNote:(most,least)=>`<b>${most.label}</b> is the most opinionated model — it pushes probabilities furthest from a coin-flip. <b>${least.label}</b> is by far the most cautious: an average winner-gap of just ${least.mean_decisiveness.toFixed(3)} and a top-outcome of only ${pct(least.mean_top_outcome)} means it rates almost every match as a near-even three-way toss-up. That hedging is also why ${least.label} appears to "love" minnows and "fear" giants in the previous section — when everyone is flattened toward 33/33/33, the favourites lose the most ground relative to the consensus.`,
  findCards:()=>[
    ['Open at the top', `${tn(D.title_odds[0].team)}, ${tn(D.title_odds[1].team)} and ${tn(D.title_odds[2].team)} are within ${((D.title_odds[0].prob-D.title_odds[2].prob)*100).toFixed(1)} pts of each other.`],
    ['Chalk vs chaos', `The on-paper favourite wins only ${pct(D.title_odds[0].prob)} of simulations — the field takes the other ${pct(1-D.title_odds[0].prob)}.`],
    ['Most likely Final', `${tn(D.final_pairs[0].pair[0])} vs ${tn(D.final_pairs[0].pair[1])} is the single most probable Final, but only ${pct(D.final_pairs[0].prob)} of the time.`],
    ['Biggest split', `Models disagree most on ${tn(D.disagreements[0].pair[0])} vs ${tn(D.disagreements[0].pair[1])} — a ${(D.disagreements[0].spread*100).toFixed(0)}-pt spread in win probability.`],
  ],
  S:{
   kicker:"FIFA World Cup 2026 · Consensus Forecast",
   h1:'The <span class="hl">AI Consensus</span> of the World Cup',
   lead:'A Monte-Carlo forecast of the 2026 World Cup built from the equal-weight consensus of five frontier AI models, then stress-tested across <b id="hSims"></b> simulated tournaments.',
   nav_odds:'Title odds', nav_stages:'Stage reach', nav_groups:'Groups', nav_bracket:'Predicted bracket',
   nav_bias:'Model biases', nav_conf:'Confidence', nav_find:'Findings', nav_expl:'Team explorer',
   e_odds:'01 · Who wins', t_odds:'Championship probabilities',
   s_odds:'Share of simulated tournaments each team lifted the trophy. The top three are separated by barely two percentage points — a genuinely open field at the top.',
   e_stages:'02 · How far', t_stages:'Probability of reaching each round',
   s_stages:'Each cell is the chance a team <b>reaches</b> (plays in) that round. <b>Reaching the group stage is 100% for all 48 teams</b>, so it is omitted. The knockout columns are cumulative — reaching a round implies winning the previous one. The final column, <b>Win title</b>, is the only one that means <b>winning</b> rather than reaching: note how it is always lower than <b>Reach Final</b>, since a team can play the final and lose it. Cell shading scales with probability.',
   e_groups:'03 · Group stage', t_groups:'Group predictions',
   s_groups:"Per group: each team's probability of advancing to the knockouts (bar) and of finishing top of the group (number).",
   e_bracket:'04 · The prediction', t_bracket:'The chalk bracket',
   s_bracket:'A single best-guess tournament where the favourite wins every match (knockout win-probability shown). This is the modal "if everything goes to form" path — reality will diverge, which is exactly what the odds above quantify.',
   e_bias:'05 · Model personalities', t_bias:'Who each model loves &amp; fears',
   s_bias:'Each model\'s team rating minus the consensus, then centred to remove overall calibration bias. <b style="color:var(--color-success)">Green</b> = teams this model inflates relative to the field (its pet favourites). <b style="color:var(--color-error)">Red</b> = teams it suppresses (its blind spots).',
   e_conf:'06 · Conviction', t_conf:'How confident is each model?',
   s_conf:'Decisiveness = how far a model pushes its probabilities away from a 33/33/33 coin-flip. A high score means strong opinions; a low score means hedging.',
   e_find:'07 · The fun stuff', t_find:'Things worth noticing',
   find_dis_h:'Biggest model disagreements', find_dis_p:'Single matchups where the stronger side\'s win prob varied most across models.',
   find_coin_h:'Truest coin-flips (consensus)', find_coin_p:'Matchups the consensus sees as most evenly balanced.',
   find_finals_h:'Most likely Final pairings', find_lop_h:'Most lopsided matchups',
   e_expl:'08 · Deep dive', t_expl:'Team explorer',
   s_expl:'Pick any of the 48 teams for a full breakdown: stage probabilities, who knocks them out most, and their most common tournament journey.',
   foot1:'Generated from the consensus probabilities (equal-weight average of Gemini Pro 3.1, GPT-5, Grok 4.2, Claude Opus 4.6, Claude Sonnet 4.6). <span id="footMeta"></span>',
   foot2:'Knockout matches resolve draws by splitting the draw probability between the two sides. Stage probabilities are monotonic (reaching a round implies reaching all earlier rounds).',
  },
 },
 es:{
  title:"El consenso de las IA del Mundial — Pronóstico 2026",
  REACH:{ko:'16avos',r16:'Octavos',qf:'Cuartos',sf:'Semis',final:'Final',champion:'Título'},
  roundLabels:{r32:'16avos',r16:'Octavos',qf:'Cuartos',sf:'Semis',final:'Final'},
  exitRound:{ko:'dieciseisavos',r16:'octavos',qf:'cuartos',sf:'semifinales',final:'la final'},
  championWord:'Campeón',
  jChampion:'Gana el torneo',
  jGroup:p=>`${p} de grupo → eliminada en fase de grupos`,
  jKO:(p,r,e)=>`${p} de grupo → eliminada en ${r} (por ${e})`,
  favourite:'Favorito', chalkChampion:'Campeón sobre el papel', simulationsWord:'simulaciones',
  modelsWord:'modelos', seedWord:'semilla', vs:'vs',
  th_team:'Selección', th_wingroup:'Gana grupo', groupWord:'GRUPO',
  groupsFoot:(teamHtml,p)=>`barra = avanza · 1.ª: <b style="color:var(--accent)">${teamHtml}</b> gana el grupo ${p}`,
  topFavs:'Favoritas:', inflates:'Infla ▲', suppresses:'Minusvalora ▼',
  confHeaders:['Modelo','Convicción','Resultado top medio','Empate medio %','|gana₁−gana₂|'],
  findHeaders:{matchup:'Partido',spread:'Horquilla',range:'Rango',win1:'Gana₁',draw:'Empate',win2:'Gana₂',pairing:'Emparejamiento',prob:'Probabilidad',fav:'Favorito'},
  stat:{winCup:'Gana el Mundial',reachKO:'Llega a eliminatorias',winGroup:'Gana el grupo',topElim:'Mayor verdugo'},
  expl:{reachTitle:'Probabilidad de llegar a cada ronda',
    reachSub:'"Llega" = juega esa ronda. "Título" = levanta el trofeo.',
    groupFinish:'Posición de grupo', howOut:'Cómo cae eliminada',
    commonJourneys:'Recorridos más habituales',
    journeysSub:n=>`Las historias del torneo que más se repiten en ${n} simulaciones.`,
    mostRepeated:'Resultado individual más repetido:'},
  rankLabels:['Gana grupo','2.º','3.º','4.º'],
  elimGroupStage:'Eliminada en fase de grupos', knockedOutBy:'Eliminada por',
  chalkNote:()=>`En el cuadro sobre el papel el favorito siempre avanza, así que corona a <b>${tn(D.chalk.champion)}</b>. Pero en ${nfmt(D.meta.n_sims)} simulaciones aleatorias el campeón más frecuente es <b>${tn(D.most_common_champion)}</b> con solo ${pct(D.title_odds[0].prob)} — la prueba de que lo "más probable sobre el papel" pierde mucho más de lo que gana.`,
  confNote:(most,least)=>`<b>${most.label}</b> es el modelo con más convicción: empuja las probabilidades más lejos de un 33/33/33. <b>${least.label}</b> es con diferencia el más cauto: una diferencia media entre ganadores de apenas ${least.mean_decisiveness.toFixed(3)} y un resultado más probable de solo ${pct(least.mean_top_outcome)} significan que ve casi todos los partidos como un triple empate. Ese exceso de cautela es también por lo que ${least.label} parece "amar" a las cenicientas y "temer" a los gigantes en la sección anterior — cuando todo se aplana hacia 33/33/33, los favoritos pierden más terreno respecto al consenso.`,
  findCards:()=>[
    ['Apretado arriba', `${tn(D.title_odds[0].team)}, ${tn(D.title_odds[1].team)} y ${tn(D.title_odds[2].team)} están a menos de ${((D.title_odds[0].prob-D.title_odds[2].prob)*100).toFixed(1)} puntos entre sí.`],
    ['Papel vs caos', `El favorito sobre el papel gana solo el ${pct(D.title_odds[0].prob)} de las simulaciones — el resto se reparte el otro ${pct(1-D.title_odds[0].prob)}.`],
    ['Final más probable', `${tn(D.final_pairs[0].pair[0])} vs ${tn(D.final_pairs[0].pair[1])} es la final más probable, pero solo el ${pct(D.final_pairs[0].prob)} de las veces.`],
    ['Mayor desacuerdo', `Los modelos discrepan más en ${tn(D.disagreements[0].pair[0])} vs ${tn(D.disagreements[0].pair[1])} — una horquilla de ${(D.disagreements[0].spread*100).toFixed(0)} puntos en la probabilidad de victoria.`],
  ],
  S:{
   kicker:"Mundial FIFA 2026 · Pronóstico de consenso",
   h1:'El <span class="hl">consenso de las IA</span> del Mundial',
   lead:'Un pronóstico Monte-Carlo del Mundial 2026 construido a partir del consenso equiponderado de cinco modelos de IA punteros, y sometido a <b id="hSims"></b> torneos simulados.',
   nav_odds:'Título', nav_stages:'Rondas', nav_groups:'Grupos', nav_bracket:'Cuadro',
   nav_bias:'Sesgos', nav_conf:'Convicción', nav_find:'Curiosidades', nav_expl:'Explorador',
   e_odds:'01 · Quién gana', t_odds:'Probabilidades de título',
   s_odds:'Porcentaje de torneos simulados en los que cada selección levantó el trofeo. Los tres primeros están separados por apenas dos puntos porcentuales — un grupo de cabeza realmente abierto.',
   e_stages:'02 · Hasta dónde', t_stages:'Probabilidad de llegar a cada ronda',
   s_stages:'Cada celda es la probabilidad de que una selección <b>llegue</b> (juegue) a esa ronda. <b>Llegar a la fase de grupos es 100% para las 48 selecciones</b>, así que se omite. Las columnas eliminatorias son acumulativas — llegar a una ronda implica haber ganado la anterior. La última columna, <b>Título</b>, es la única que significa <b>ganar</b> y no llegar: fíjate en que siempre es menor que <b>Final</b>, porque una selección puede jugar la final y perderla. El sombreado escala con la probabilidad.',
   e_groups:'03 · Fase de grupos', t_groups:'Pronóstico de grupos',
   s_groups:'Por grupo: la probabilidad de cada selección de avanzar a las eliminatorias (barra) y de terminar primera de grupo (número).',
   e_bracket:'04 · El pronóstico', t_bracket:'El cuadro sobre el papel',
   s_bracket:'Un único pronóstico donde el favorito gana todos los partidos (se muestra la probabilidad de victoria en eliminatorias). Es el camino modal de "si todo sale según lo previsto" — la realidad divergirá, que es justo lo que cuantifican las probabilidades de arriba.',
   e_bias:'05 · Personalidad de los modelos', t_bias:'A quién ama y teme cada modelo',
   s_bias:'La valoración de cada modelo menos el consenso, centrada para eliminar el sesgo de calibración general. <b style="color:var(--color-success)">Verde</b> = selecciones que este modelo infla respecto al resto (sus favoritas). <b style="color:var(--color-error)">Rojo</b> = selecciones que minusvalora (sus puntos ciegos).',
   e_conf:'06 · Convicción', t_conf:'¿Cuánta convicción tiene cada modelo?',
   s_conf:'La convicción mide cuánto aleja un modelo sus probabilidades de un 33/33/33. Una puntuación alta significa opiniones firmes; una baja, prudencia.',
   e_find:'07 · Lo curioso', t_find:'Cosas que merece la pena ver',
   find_dis_h:'Mayores desacuerdos entre modelos', find_dis_p:'Partidos donde más varió la probabilidad de victoria del lado más fuerte.',
   find_coin_h:'Partidos más igualados (consenso)', find_coin_p:'Partidos que el consenso ve más equilibrados.',
   find_finals_h:'Finales más probables', find_lop_h:'Partidos más desiguales',
   e_expl:'08 · Análisis', t_expl:'Explorador de selecciones',
   s_expl:'Elige cualquiera de las 48 selecciones para ver el desglose completo: probabilidades por ronda, quién la elimina más y su recorrido más habitual.',
   foot1:'Generado a partir de las probabilidades de consenso (media equiponderada de Gemini Pro 3.1, GPT-5, Grok 4.2, Claude Opus 4.6, Claude Sonnet 4.6). <span id="footMeta"></span>',
   foot2:'Los partidos de eliminatoria reparten la probabilidad de empate entre los dos equipos. Las probabilidades por ronda son monótonas (llegar a una ronda implica haber llegado a todas las anteriores).',
  },
 },
};

// ── language state ───────────────────────────────────────────────────────────
let LANG = localStorage.getItem('wc_lang');
if(LANG!=='en' && LANG!=='es')
  LANG = (navigator.language||'').toLowerCase().startsWith('es') ? 'es' : 'en';

const R = () => I18N[LANG];
function fmtJourney(j){
  const r=R();
  if(j.stage==='champion') return r.jChampion;
  if(j.stage==='group') return r.jGroup(placeWord(j.rank));
  return r.jKO(placeWord(j.rank), r.exitRound[j.stage], tn(j.eliminator));
}

let currentTeam = D.title_odds[0].team;

// ── renderers ────────────────────────────────────────────────────────────────
function setStatic(){
  const S=R().S;
  document.documentElement.lang=LANG;
  document.title=R().title;
  for(const id in S){ const e=document.getElementById(id); if(e) e.innerHTML=S[id]; }
  const hs=document.getElementById('hSims'); if(hs) hs.textContent=nfmt(D.meta.n_sims);
}

function renderHero(){
  const r=R(), top1=D.title_odds[0];
  document.getElementById('heroMeta').innerHTML=[
    `<span class="chip">${r.favourite}&nbsp;<b>${tn(top1.team)} · ${pct(top1.prob)}</b></span>`,
    `<span class="chip">${r.chalkChampion}&nbsp;<b>${tn(D.chalk.champion)}</b></span>`,
    `<span class="chip"><b>${nfmt(D.meta.n_sims)}</b>&nbsp;${r.simulationsWord}</span>`,
    `<span class="chip"><b>5</b>&nbsp;${r.modelsWord} · ${r.seedWord} ${D.meta.seed}</span>`,
  ].join('');
  const fm=document.getElementById('footMeta');
  if(fm) fm.textContent=`${nfmt(D.meta.n_sims)} ${r.simulationsWord} · ${r.seedWord} ${D.meta.seed}.`;
}

function renderOdds(){
  const box=document.getElementById('oddsBars'); box.innerHTML='';
  const max=D.title_odds[0].prob;
  D.title_odds.slice(0,22).forEach(o=>{
    box.appendChild(el('div','bar-row',
      `<div class="name">${tn(o.team)}</div>
       <div class="bar-track"><div class="bar-fill" style="width:${(o.prob/max*100).toFixed(1)}%"></div></div>
       <div class="val">${pct(o.prob)}</div>`));
  });
}

function renderStage(){
  const r=R(), stages=D.meta.stages, t=document.getElementById('stageTable');
  let head=`<thead><tr><th>${r.th_team}</th><th class="num">${r.th_wingroup}</th>`+
    stages.map(s=>`<th class="num">${r.REACH[s]}</th>`).join('')+'</tr></thead>';
  const rows=D.title_odds.map(o=>D.teams[o.team]);
  let body='<tbody>'+rows.map(tm=>
    `<tr><td>${tn(tm.team)} <span class="dim">${tm.group}</span></td>`+
    `<td class="num" style="${heat(tm.group_win_prob)}">${pct0(tm.group_win_prob)}</td>`+
    stages.map(s=>`<td class="num" style="${heat(tm.stage_prob[s])}">${pct0(tm.stage_prob[s])}</td>`).join('')+
    `</tr>`).join('')+'</tbody>';
  t.innerHTML=head+body;
}

function renderGroups(){
  const r=R(), grid=document.getElementById('groupGrid'); grid.innerHTML='';
  Object.keys(D.groups).forEach(g=>{
    let h=`<div class="gl">${r.groupWord} ${g}</div>`;
    D.groups[g].forEach(row=>{
      h+=`<div class="team-line"><div class="tn">${tn(row.team)}</div>
        <div style="display:flex;align-items:center;gap:8px">
          <div class="mini-track" style="flex:1"><div class="mini-fill" style="width:${(row.advance*100).toFixed(0)}%"></div></div>
          <span class="pct">${pct0(row.advance)}</span></div></div>`;
    });
    h+=`<div class="dim" style="font-size:.7rem;margin-top:8px">${r.groupsFoot(tn(D.groups[g][0].team),pct0(D.groups[g][0].win_group))}</div>`;
    grid.appendChild(el('div','card group-card',h));
  });
}

function renderBracket(){
  const r=R(), box=document.getElementById('bracketBox'); box.innerHTML='';
  [['r32',r.roundLabels.r32],['r16',r.roundLabels.r16],['qf',r.roundLabels.qf],
   ['sf',r.roundLabels.sf],['final',r.roundLabels.final]].forEach(([key,label])=>{
    const col=el('div','round'); col.appendChild(el('h4',null,label));
    D.chalk[key].forEach(m=>{
      const w1=m.winner===m.t1, w2=m.winner===m.t2;
      col.appendChild(el('div','match',
        `<div class="t ${w1?'win':''}"><span>${tn(m.t1)}</span><span class="p">${w1?pct0(m.p):''}</span></div>
         <div class="t ${w2?'win':''}"><span>${tn(m.t2)}</span><span class="p">${w2?pct0(m.p):''}</span></div>`));
    });
    box.appendChild(col);
  });
  const champCol=el('div','round');
  champCol.appendChild(el('h4',null,r.championWord));
  champCol.appendChild(el('div','champ-box','🏆 '+tn(D.chalk.champion)));
  box.appendChild(champCol);
  document.getElementById('chalkNote').innerHTML=r.chalkNote();
}

function renderBias(){
  const r=R(), box=document.getElementById('biasBox'); box.innerHTML='';
  const order=['gemini_pro_31','gpt_52','grok_42','opus_46','sonnet_46'];
  const maxAbs=Math.max(...order.flatMap(m=>D.model_bias[m].loved.concat(D.model_bias[m].dreaded).map(x=>Math.abs(x.delta))));
  order.forEach(m=>{
    const b=D.model_bias[m];
    const mk=arr=>arr.map(x=>{
      const w=(Math.abs(x.delta)/maxAbs*50).toFixed(1), pos=x.delta>=0;
      return `<div class="delta-row"><div class="name">${tn(x.team)}</div>
        <div class="delta-track"><div class="zero"></div>
          <div class="${pos?'delta-pos':'delta-neg'}" style="width:${w}%"></div></div>
        <div class="dv" style="color:${pos?'var(--color-success)':'var(--color-error)'}">${pos?'+':''}${x.delta.toFixed(3)}</div></div>`;
    }).join('');
    box.appendChild(el('div','card',
      `<h4>${b.label}</h4>
       <p class="dim" style="font-size:.78rem;margin:2px 0 6px">${r.topFavs} ${b.top_favorites.slice(0,4).map(f=>tn(f.team)).join(', ')}</p>
       <div class="bias-cols">
         <div><div class="eyebrow" style="color:var(--color-success)">${r.inflates}</div>${mk(b.loved)}</div>
         <div><div class="eyebrow" style="color:var(--color-error)">${r.suppresses}</div>${mk(b.dreaded)}</div>
       </div>`)).style.marginBottom='16px';
  });
}

function renderConf(){
  const r=R(), t=document.getElementById('confTable'), c=D.model_confidence;
  const maxd=Math.max(...c.map(x=>x.mean_dev_from_uniform)), H=r.confHeaders;
  t.innerHTML=`<thead><tr><th>${H[0]}</th><th class="num">${H[1]}</th><th class="num">${H[2]}</th>
    <th class="num">${H[3]}</th><th class="num">${H[4]}</th></tr></thead><tbody>`+
    c.map((x,i)=>`<tr><td>${i+1}. ${x.label}</td>
      <td class="num"><div style="display:flex;align-items:center;gap:8px;justify-content:flex-end">
        <div class="mini-track" style="width:80px"><div class="mini-fill" style="width:${(x.mean_dev_from_uniform/maxd*100).toFixed(0)}%"></div></div>
        ${x.mean_dev_from_uniform.toFixed(3)}</div></td>
      <td class="num">${pct(x.mean_top_outcome)}</td>
      <td class="num">${pct(x.mean_draw)}</td>
      <td class="num">${x.mean_decisiveness.toFixed(3)}</td></tr>`).join('')+'</tbody>';
  document.getElementById('confNote').innerHTML=r.confNote(c[0],c[c.length-1]);
}

function renderFindings(){
  const r=R(), H=r.findHeaders, fg=document.getElementById('findGrid'); fg.innerHTML='';
  r.findCards().forEach(([h,p])=>fg.appendChild(el('div','card',
    `<h4 style="color:var(--accent);font-size:1rem">${h}</h4><p class="muted" style="margin-top:6px;font-size:.9rem">${p}</p>`)));

  document.getElementById('disagreeTable').innerHTML=
    `<thead><tr><th>${H.matchup}</th><th class="num">${H.spread}</th><th class="num">${H.range}</th></tr></thead><tbody>`+
    D.disagreements.slice(0,6).map(d=>{const v=Object.values(d.by_model);
      return `<tr><td>${tn(d.pair[0])} ${r.vs} ${tn(d.pair[1])}</td><td class="num">${(d.spread*100).toFixed(0)}pt</td>
        <td class="num">${pct0(Math.min(...v))}–${pct0(Math.max(...v))}</td></tr>`;}).join('')+'</tbody>';

  document.getElementById('coinTable').innerHTML=
    `<thead><tr><th>${H.matchup}</th><th class="num">${H.win1}</th><th class="num">${H.draw}</th><th class="num">${H.win2}</th></tr></thead><tbody>`+
    D.coin_flips.slice(0,6).map(c=>`<tr><td>${tn(c.t1)} ${r.vs} ${tn(c.t2)}</td><td class="num">${pct0(c.p1)}</td>
      <td class="num">${pct0(c.draw)}</td><td class="num">${pct0(c.p2)}</td></tr>`).join('')+'</tbody>';

  document.getElementById('finalsTable').innerHTML=
    `<thead><tr><th>${H.pairing}</th><th class="num">${H.prob}</th></tr></thead><tbody>`+
    D.final_pairs.slice(0,7).map(f=>`<tr><td>${tn(f.pair[0])} ${r.vs} ${tn(f.pair[1])}</td><td class="num">${pct(f.prob)}</td></tr>`).join('')+'</tbody>';

  document.getElementById('lopTable').innerHTML=
    `<thead><tr><th>${H.matchup}</th><th class="num">${H.fav}</th></tr></thead><tbody>`+
    D.lopsided.slice(0,6).map(c=>{const f=c.p1>=c.p2?[c.t1,c.p1,c.t2]:[c.t2,c.p2,c.t1];
      return `<tr><td>${tn(f[0])} ${r.vs} ${tn(f[2])}</td><td class="num">${pct0(f[1])}</td></tr>`;}).join('')+'</tbody>';
}

function renderExplorer(){
  const r=R(), sel=document.getElementById('teamSelect');
  sel.innerHTML='';
  D.title_odds.map(o=>o.team).forEach(t=>{
    const o=el('option'); o.value=t; o.textContent=tn(t);
    if(t===currentTeam) o.selected=true;
    sel.appendChild(o);
  });
  const tm=D.teams[currentTeam], stages=D.meta.stages;
  document.getElementById('teamGroupPill').textContent=`${r.groupWord==='GRUPO'?'Grupo':'Group'} ${tm.group}`;
  const stageBars=stages.map(s=>
    `<div class="bar-row"><div class="name">${r.REACH[s]}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(tm.stage_prob[s]*100).toFixed(1)}%"></div></div>
      <div class="val">${pct(tm.stage_prob[s])}</div></div>`).join('');
  const ranks=[1,2,3,4].map(k=>`<div class="kv"><span class="k">${r.rankLabels[k-1]}</span><span class="v">${pct(tm.group_rank_dist[String(k)])}</span></div>`).join('');
  const elim=tm.eliminator_breakdown.map(e=>`<div class="kv"><span class="k">${e.by==='(Group stage)'?r.elimGroupStage:r.knockedOutBy+' '+tn(e.by)}</span><span class="v">${pct(e.prob)}</span></div>`).join('');
  const journeys=tm.top_journeys.map(j=>`<div class="journey"><span>${fmtJourney(j)}</span><span class="jp">${pct(j.prob)}</span></div>`).join('');
  const elimTeam=tm.top_eliminator?tn(tm.top_eliminator.team):'—';
  document.getElementById('teamPanel').innerHTML=`
    <div class="stat-grid">
      <div class="stat"><div class="v">${pct(tm.champion_prob)}</div><div class="l">${r.stat.winCup}</div></div>
      <div class="stat"><div class="v">${pct(tm.stage_prob.ko)}</div><div class="l">${r.stat.reachKO}</div></div>
      <div class="stat"><div class="v">${pct(tm.group_win_prob)}</div><div class="l">${r.stat.winGroup}</div></div>
      <div class="stat"><div class="v">${elimTeam}</div><div class="l">${r.stat.topElim}</div></div>
    </div>
    <div class="grid g2">
      <div class="card"><h4 style="margin-bottom:2px">${r.expl.reachTitle}</h4>
        <p class="dim" style="font-size:.76rem;margin-bottom:10px">${r.expl.reachSub}</p>${stageBars}</div>
      <div class="card"><h4 style="margin-bottom:10px">${r.expl.groupFinish}</h4>${ranks}
        <h4 style="margin:16px 0 10px">${r.expl.howOut}</h4>${elim}</div>
    </div>
    <div class="card" style="margin-top:16px"><h4 style="margin-bottom:4px">${r.expl.commonJourneys}</h4>
      <p class="dim" style="font-size:.8rem;margin-bottom:10px">${r.expl.journeysSub(nfmt(D.meta.n_sims))}</p>
      ${journeys}
      <div class="note" style="margin-top:14px">${r.expl.mostRepeated} <b>${fmtJourney(tm.modal_journey)}</b> (${pct(tm.modal_journey.prob)}).</div>
    </div>`;
}

function renderAll(){
  setStatic(); renderHero(); renderOdds(); renderStage(); renderGroups();
  renderBracket(); renderBias(); renderConf(); renderFindings(); renderExplorer();
  document.querySelectorAll('.lang-toggle button').forEach(b=>b.classList.toggle('active', b.dataset.lang===LANG));
}

document.getElementById('teamSelect').addEventListener('change',e=>{currentTeam=e.target.value; renderExplorer();});
document.querySelectorAll('.lang-toggle button').forEach(b=>b.addEventListener('click',()=>{
  LANG=b.dataset.lang; localStorage.setItem('wc_lang',LANG); renderAll();
}));

renderAll();
</script>
</body>
</html>"""

html = HTML.replace("__DATA__", DATA_JSON)
for path in (OUT, WEB_OUT):
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {path} ({len(html)//1024} KB)")
