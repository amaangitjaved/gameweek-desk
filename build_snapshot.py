"""Builds the offline demo snapshot: data/snapshot_2026_27.json.

Why a snapshot exists at all
----------------------------
Two reasons, both of which are part of the system design rather than a
shortcut:

1. Determinism for review. A reviewer clicking the link should see the same
   queue the video showed. A live API means the demo changes under you.
2. Pre-season reality. In early August the FPL API returns a season where
   every player has zero points, zero minutes and zero ICT. Any model that
   trains on current-season totals collapses to a constant. The snapshot
   carries prior-season per-90 priors, which is exactly what the live system
   would need in August anyway.

The player -> club -> position -> price mapping is illustrative and may not
match the live game after transfer activity. Prior-season rates are
synthesised deterministically from price and position, which is a reasonable
proxy because FPL price is itself set from expected output. This is stated in
the in-app Assumptions page.
"""

from __future__ import annotations

import json
import os
import random

from core.config import DATA_DIR, SNAPSHOT_PATH

SEED = 20262027

# (web_name, first, second, team, element_type, price_millions)
# element_type: 1=GK 2=DEF 3=MID 4=FWD
ROSTER: list[tuple[str, str, str, str, int, float]] = [
    # Arsenal
    ("Raya", "David", "Raya", "Arsenal", 1, 5.5),
    ("Saliba", "William", "Saliba", "Arsenal", 2, 6.0),
    ("Gabriel", "Gabriel", "Magalhaes", "Arsenal", 2, 6.2),
    ("Timber", "Jurrien", "Timber", "Arsenal", 2, 5.6),
    ("Saka", "Bukayo", "Saka", "Arsenal", 3, 10.0),
    ("Odegaard", "Martin", "Odegaard", "Arsenal", 3, 8.3),
    ("Rice", "Declan", "Rice", "Arsenal", 3, 6.5),
    ("Havertz", "Kai", "Havertz", "Arsenal", 4, 7.8),
    # Aston Villa
    ("Martinez", "Emiliano", "Martinez", "Aston Villa", 1, 5.0),
    ("Konsa", "Ezri", "Konsa", "Aston Villa", 2, 4.5),
    ("Digne", "Lucas", "Digne", "Aston Villa", 2, 4.7),
    ("Rogers", "Morgan", "Rogers", "Aston Villa", 3, 7.0),
    ("McGinn", "John", "McGinn", "Aston Villa", 3, 5.5),
    ("Tielemans", "Youri", "Tielemans", "Aston Villa", 3, 5.6),
    ("Watkins", "Ollie", "Watkins", "Aston Villa", 4, 9.0),
    ("Malen", "Donyell", "Malen", "Aston Villa", 4, 5.6),
    # Bournemouth
    ("Petrovic", "Djordje", "Petrovic", "Bournemouth", 1, 4.5),
    ("Senesi", "Marcos", "Senesi", "Bournemouth", 2, 4.8),
    ("Truffert", "Adrien", "Truffert", "Bournemouth", 2, 4.5),
    ("Semenyo", "Antoine", "Semenyo", "Bournemouth", 3, 7.5),
    ("Tavernier", "Marcus", "Tavernier", "Bournemouth", 3, 5.2),
    ("Scott", "Alex", "Scott", "Bournemouth", 3, 4.9),
    ("Evanilson", "Evanilson", "Evanilson", "Bournemouth", 4, 6.8),
    ("Kluivert", "Justin", "Kluivert", "Bournemouth", 3, 6.4),
    # Brentford
    ("Kelleher", "Caoimhin", "Kelleher", "Brentford", 1, 4.6),
    ("Collins", "Nathan", "Collins", "Brentford", 2, 5.0),
    ("Ajer", "Kristoffer", "Ajer", "Brentford", 2, 4.4),
    ("Henderson", "Jordan", "Henderson", "Brentford", 3, 5.0),
    ("Damsgaard", "Mikkel", "Damsgaard", "Brentford", 3, 6.0),
    ("Schade", "Kevin", "Schade", "Brentford", 3, 6.2),
    ("Wissa", "Yoane", "Wissa", "Brentford", 4, 7.2),
    ("Thiago", "Igor", "Thiago", "Brentford", 4, 6.0),
    # Brighton
    ("Verbruggen", "Bart", "Verbruggen", "Brighton", 1, 4.7),
    ("Van Hecke", "Jan Paul", "van Hecke", "Brighton", 2, 4.8),
    ("Estupinan", "Pervis", "Estupinan", "Brighton", 2, 5.0),
    ("Mitoma", "Kaoru", "Mitoma", "Brighton", 3, 6.6),
    ("Minteh", "Yankuba", "Minteh", "Brighton", 3, 6.1),
    ("Gruda", "Brajan", "Gruda", "Brighton", 3, 5.3),
    ("Welbeck", "Danny", "Welbeck", "Brighton", 4, 6.4),
    ("Joao Pedro", "Joao", "Pedro", "Brighton", 4, 7.6),
    # Burnley
    ("Trafford", "James", "Trafford", "Burnley", 1, 4.4),
    ("Esteve", "Maxime", "Esteve", "Burnley", 2, 4.2),
    ("Roberts", "Connor", "Roberts", "Burnley", 2, 4.3),
    ("Cullen", "Josh", "Cullen", "Burnley", 3, 4.9),
    ("Anthony", "Jaidon", "Anthony", "Burnley", 3, 5.4),
    ("Brownhill", "Josh", "Brownhill", "Burnley", 3, 5.1),
    ("Flemming", "Zian", "Flemming", "Burnley", 4, 5.3),
    ("Foster", "Lyle", "Foster", "Burnley", 4, 4.9),
    # Chelsea
    ("Sanchez", "Robert", "Sanchez", "Chelsea", 1, 4.9),
    ("Cucurella", "Marc", "Cucurella", "Chelsea", 2, 6.0),
    ("James", "Reece", "James", "Chelsea", 2, 5.4),
    ("Colwill", "Levi", "Colwill", "Chelsea", 2, 5.0),
    ("Palmer", "Cole", "Palmer", "Chelsea", 3, 10.5),
    ("Neto", "Pedro", "Neto", "Chelsea", 3, 7.0),
    ("Enzo", "Enzo", "Fernandez", "Chelsea", 3, 6.6),
    ("Joao Pedro C", "Joao Pedro", "Cardoso", "Chelsea", 4, 7.5),
    # Crystal Palace
    ("Henderson", "Dean", "Henderson", "Crystal Palace", 1, 5.0),
    ("Guehi", "Marc", "Guehi", "Crystal Palace", 2, 4.9),
    ("Mitchell", "Tyrick", "Mitchell", "Crystal Palace", 2, 4.8),
    ("Munoz", "Daniel", "Munoz", "Crystal Palace", 2, 5.6),
    ("Eze", "Eberechi", "Eze", "Crystal Palace", 3, 7.6),
    ("Sarr", "Ismaila", "Sarr", "Crystal Palace", 3, 6.5),
    ("Mateta", "Jean-Philippe", "Mateta", "Crystal Palace", 4, 7.7),
    ("Nketiah", "Eddie", "Nketiah", "Crystal Palace", 4, 5.4),
    # Everton
    ("Pickford", "Jordan", "Pickford", "Everton", 1, 5.3),
    ("Tarkowski", "James", "Tarkowski", "Everton", 2, 5.1),
    ("Mykolenko", "Vitalii", "Mykolenko", "Everton", 2, 4.4),
    ("O'Brien", "Jake", "O'Brien", "Everton", 2, 4.5),
    ("Garner", "James", "Garner", "Everton", 3, 5.2),
    ("Ndiaye", "Iliman", "Ndiaye", "Everton", 3, 6.6),
    ("Grealish", "Jack", "Grealish", "Everton", 3, 6.8),
    ("Beto", "Beto", "Beto", "Everton", 4, 5.5),
    # Fulham
    ("Leno", "Bernd", "Leno", "Fulham", 1, 5.0),
    ("Robinson", "Antonee", "Robinson", "Fulham", 2, 5.2),
    ("Andersen", "Joachim", "Andersen", "Fulham", 2, 4.7),
    ("Iwobi", "Alex", "Iwobi", "Fulham", 3, 6.3),
    ("Wilson", "Harry", "Wilson", "Fulham", 3, 5.6),
    ("Smith Rowe", "Emile", "Smith Rowe", "Fulham", 3, 5.4),
    ("Jimenez", "Raul", "Jimenez", "Fulham", 4, 6.5),
    ("Muniz", "Rodrigo", "Muniz", "Fulham", 4, 5.6),
    # Leeds
    ("Darlow", "Karl", "Darlow", "Leeds", 1, 4.3),
    ("Rodon", "Joe", "Rodon", "Leeds", 2, 4.2),
    ("Bogle", "Jayden", "Bogle", "Leeds", 2, 4.4),
    ("Gruev", "Ilia", "Gruev", "Leeds", 3, 4.7),
    ("Aaronson", "Brenden", "Aaronson", "Leeds", 3, 5.4),
    ("Gnonto", "Wilfried", "Gnonto", "Leeds", 3, 5.6),
    ("Piroe", "Joel", "Piroe", "Leeds", 4, 5.7),
    ("Nmecha", "Lukas", "Nmecha", "Leeds", 4, 5.0),
    # Liverpool
    ("Alisson", "Alisson", "Becker", "Liverpool", 1, 5.6),
    ("Van Dijk", "Virgil", "van Dijk", "Liverpool", 2, 6.2),
    ("Konate", "Ibrahima", "Konate", "Liverpool", 2, 5.4),
    ("Robertson", "Andrew", "Robertson", "Liverpool", 2, 5.8),
    ("Salah", "Mohamed", "Salah", "Liverpool", 3, 14.2),
    ("Szoboszlai", "Dominik", "Szoboszlai", "Liverpool", 3, 6.7),
    ("Gakpo", "Cody", "Gakpo", "Liverpool", 3, 7.6),
    ("Ekitike", "Hugo", "Ekitike", "Liverpool", 4, 8.6),
    # Man City
    ("Ederson", "Ederson", "Moraes", "Man City", 1, 5.4),
    ("Dias", "Ruben", "Dias", "Man City", 2, 6.0),
    ("Gvardiol", "Josko", "Gvardiol", "Man City", 2, 6.1),
    ("Ait-Nouri", "Rayan", "Ait-Nouri", "Man City", 2, 5.7),
    ("Foden", "Phil", "Foden", "Man City", 3, 8.0),
    ("Reijnders", "Tijjani", "Reijnders", "Man City", 3, 6.4),
    ("Doku", "Jeremy", "Doku", "Man City", 3, 6.9),
    ("Haaland", "Erling", "Haaland", "Man City", 4, 14.8),
    # Man Utd
    ("Onana", "Andre", "Onana", "Man Utd", 1, 5.0),
    ("Martinez", "Lisandro", "Martinez", "Man Utd", 2, 5.0),
    ("Dalot", "Diogo", "Dalot", "Man Utd", 2, 5.3),
    ("Shaw", "Luke", "Shaw", "Man Utd", 2, 4.8),
    ("Bruno", "Bruno", "Fernandes", "Man Utd", 3, 9.2),
    ("Mbeumo", "Bryan", "Mbeumo", "Man Utd", 3, 8.2),
    ("Cunha", "Matheus", "Cunha", "Man Utd", 3, 7.9),
    ("Sesko", "Benjamin", "Sesko", "Man Utd", 4, 7.6),
    # Newcastle
    ("Pope", "Nick", "Pope", "Newcastle", 1, 5.1),
    ("Burn", "Dan", "Burn", "Newcastle", 2, 4.6),
    ("Trippier", "Kieran", "Trippier", "Newcastle", 2, 5.2),
    ("Livramento", "Tino", "Livramento", "Newcastle", 2, 5.1),
    ("Bruno G", "Bruno", "Guimaraes", "Newcastle", 3, 6.5),
    ("Barnes", "Harvey", "Barnes", "Newcastle", 3, 6.6),
    ("Murphy", "Jacob", "Murphy", "Newcastle", 3, 6.3),
    ("Isak", "Alexander", "Isak", "Newcastle", 4, 10.4),
    # Nott'm Forest
    ("Sels", "Matz", "Sels", "Nott'm Forest", 1, 5.0),
    ("Milenkovic", "Nikola", "Milenkovic", "Nott'm Forest", 2, 5.4),
    ("Murillo", "Murillo", "Murillo", "Nott'm Forest", 2, 5.3),
    ("Williams", "Neco", "Williams", "Nott'm Forest", 2, 4.9),
    ("Anderson", "Elliot", "Anderson", "Nott'm Forest", 3, 5.5),
    ("Hudson-Odoi", "Callum", "Hudson-Odoi", "Nott'm Forest", 3, 5.8),
    ("Ndoye", "Dan", "Ndoye", "Nott'm Forest", 3, 6.2),
    ("Wood", "Chris", "Wood", "Nott'm Forest", 4, 7.4),
    # Sunderland
    ("Roefs", "Robin", "Roefs", "Sunderland", 1, 4.4),
    ("Ballard", "Daniel", "Ballard", "Sunderland", 2, 4.3),
    ("Hume", "Trai", "Hume", "Sunderland", 2, 4.4),
    ("Le Fee", "Enzo", "Le Fee", "Sunderland", 3, 5.2),
    ("Xhaka", "Granit", "Xhaka", "Sunderland", 3, 5.1),
    ("Talbi", "Chemsdine", "Talbi", "Sunderland", 3, 5.0),
    ("Isidor", "Wilson", "Isidor", "Sunderland", 4, 5.5),
    ("Brobbey", "Brian", "Brobbey", "Sunderland", 4, 5.4),
    # Tottenham
    ("Vicario", "Guglielmo", "Vicario", "Tottenham", 1, 5.1),
    ("Romero", "Cristian", "Romero", "Tottenham", 2, 5.3),
    ("Porro", "Pedro", "Porro", "Tottenham", 2, 5.5),
    ("Udogie", "Destiny", "Udogie", "Tottenham", 2, 5.0),
    ("Kudus", "Mohammed", "Kudus", "Tottenham", 3, 6.8),
    ("Sarr P", "Pape Matar", "Sarr", "Tottenham", 3, 5.4),
    ("Simons", "Xavi", "Simons", "Tottenham", 3, 7.2),
    ("Richarlison", "Richarlison", "Richarlison", "Tottenham", 4, 6.6),
    # West Ham
    ("Areola", "Alphonse", "Areola", "West Ham", 1, 4.6),
    ("Kilman", "Max", "Kilman", "West Ham", 2, 4.5),
    ("Wan-Bissaka", "Aaron", "Wan-Bissaka", "West Ham", 2, 4.9),
    ("Diouf", "El Hadji", "Malick Diouf", "West Ham", 2, 4.7),
    ("Paqueta", "Lucas", "Paqueta", "West Ham", 3, 6.0),
    ("Bowen", "Jarrod", "Bowen", "West Ham", 3, 8.0),
    ("Summerville", "Crysencio", "Summerville", "West Ham", 3, 5.8),
    ("Fullkrug", "Niclas", "Fullkrug", "West Ham", 4, 5.9),
    # Wolves
    ("Johnstone", "Sam", "Johnstone", "Wolves", 1, 4.4),
    ("Agbadou", "Emmanuel", "Agbadou", "Wolves", 2, 4.4),
    ("Mosquera", "Yerson", "Mosquera", "Wolves", 2, 4.2),
    ("Ait Nouri W", "Hugo", "Bueno", "Wolves", 2, 4.3),
    ("Bellegarde", "Jean-Ricner", "Bellegarde", "Wolves", 3, 5.0),
    ("Munetsi", "Marshall", "Munetsi", "Wolves", 3, 5.3),
    ("Hwang", "Hee-Chan", "Hwang", "Wolves", 3, 5.2),
    ("Larsen", "Jorgen Strand", "Larsen", "Wolves", 4, 6.3),
]

TEAM_STRENGTH = {
    "Man City": 5, "Liverpool": 5, "Arsenal": 5, "Chelsea": 4, "Newcastle": 4,
    "Tottenham": 4, "Man Utd": 4, "Aston Villa": 4, "Brighton": 3, "Crystal Palace": 3,
    "Nott'm Forest": 3, "Bournemouth": 3, "Brentford": 3, "Fulham": 3, "Everton": 3,
    "West Ham": 2, "Wolves": 2, "Leeds": 2, "Sunderland": 2, "Burnley": 2,
}


def _per90_priors(rng: random.Random, element_type: int, price: float) -> dict:
    """Synthesise prior-season per-90 rates from price and position.

    FPL prices are themselves set from expected output, so price is a strong
    prior. Noise is seeded so the snapshot is reproducible.
    """
    base = max(0.0, price - 4.0)
    if element_type == 1:      # GK
        goals, assists, cs_rate, bonus = 0.0, 0.01 * base, 0.20 + 0.05 * base, 0.25 * base
    elif element_type == 2:    # DEF
        goals, assists, cs_rate, bonus = 0.035 * base, 0.045 * base, 0.18 + 0.045 * base, 0.22 * base
    elif element_type == 3:    # MID
        goals, assists, cs_rate, bonus = 0.075 * base, 0.075 * base, 0.10 + 0.02 * base, 0.30 * base
    else:                      # FWD
        goals, assists, cs_rate, bonus = 0.115 * base, 0.055 * base, 0.0, 0.32 * base

    jitter = lambda v, s: max(0.0, v * rng.gauss(1.0, s))  # noqa: E731

    minutes_share = min(0.96, 0.42 + 0.085 * base) * rng.gauss(1.0, 0.10)
    minutes_share = float(min(0.98, max(0.10, minutes_share)))

    return {
        "prior_goals_per90": round(jitter(goals, 0.22), 4),
        "prior_assists_per90": round(jitter(assists, 0.25), 4),
        "prior_cs_rate": round(min(0.55, jitter(cs_rate, 0.18)), 4),
        "prior_bonus_per90": round(jitter(bonus, 0.30), 4),
        "prior_minutes_share": round(minutes_share, 4),
        "prior_minutes": int(round(minutes_share * 38 * 90)),
    }


def _points_per90(element_type: int, p: dict) -> float:
    """FPL scoring applied to prior per-90 rates -> prior points per 90."""
    goal_pts = {1: 10, 2: 6, 3: 5, 4: 4}[element_type]
    cs_pts = {1: 4, 2: 4, 3: 1, 4: 0}[element_type]
    appearance = 2.0
    return (
        appearance
        + goal_pts * p["prior_goals_per90"]
        + 3.0 * p["prior_assists_per90"]
        + cs_pts * p["prior_cs_rate"]
        + p["prior_bonus_per90"]
    )


def build() -> dict:
    rng = random.Random(SEED)
    teams = sorted({r[3] for r in ROSTER})
    team_ids = {name: i + 1 for i, name in enumerate(teams)}

    players = []
    for idx, (web, first, second, team, etype, price) in enumerate(ROSTER, start=1):
        priors = _per90_priors(rng, etype, price)
        pp90 = _points_per90(etype, priors)
        players.append({
            "id": idx,
            "web_name": web,
            "first_name": first,
            "second_name": second,
            "team": team_ids[team],
            "team_name": team,
            "element_type": etype,
            "now_cost": int(round(price * 10)),
            # Current season is pre-season: these are genuinely zero, which is
            # the whole point of carrying priors.
            "total_points": 0,
            "minutes": 0,
            "ict_index": 0.0,
            "selected_by_percent": round(max(0.1, rng.gauss(6.0, 5.0)), 1),
            "prior_points_per90": round(pp90, 4),
            **priors,
        })

    # Fixture list: double round robin, difficulty from opponent strength.
    fixtures = []
    fid = 1
    n = len(teams)
    for gw in range(1, 39):
        rng.shuffle(teams)
        for i in range(0, n - 1, 2):
            h, a = teams[i], teams[i + 1]
            fixtures.append({
                "id": fid,
                "event": gw,
                "team_h": team_ids[h],
                "team_a": team_ids[a],
                "team_h_difficulty": TEAM_STRENGTH[a],
                "team_a_difficulty": min(5, TEAM_STRENGTH[h] + 1),
            })
            fid += 1

    return {
        "meta": {
            "label": "Illustrative pre-season snapshot, 2026/27",
            "synthetic": True,
            "seed": SEED,
            "note": (
                "Club and position assignments are illustrative and may not match "
                "the live game after transfer activity. Prior-season per-90 rates "
                "are synthesised deterministically from price and position."
            ),
        },
        "events": [{"id": 1, "is_current": False, "is_next": True, "name": "Gameweek 1"}]
        + [{"id": g, "is_current": False, "is_next": False, "name": f"Gameweek {g}"} for g in range(2, 39)],
        "teams": [{"id": tid, "name": name, "strength": TEAM_STRENGTH[name]}
                  for name, tid in sorted(team_ids.items(), key=lambda kv: kv[1])],
        "elements": players,
        "fixtures": fixtures,
    }


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    snap = build()
    with open(SNAPSHOT_PATH, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=1)
    print(f"wrote {SNAPSHOT_PATH}: {len(snap['elements'])} players, "
          f"{len(snap['teams'])} teams, {len(snap['fixtures'])} fixtures")
