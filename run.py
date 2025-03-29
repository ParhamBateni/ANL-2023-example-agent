import json
import time
from pathlib import Path

from utils.plot_trace import plot_trace
from utils.runners import run_session, run_tournament

RESULTS_DIR = Path("results", time.strftime('%Y%m%d-%H%M%S'))

# create results directory if it does not exist
if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True)

# Settings to run a negotiation session:
#   You need to specify the classpath of 2 agents to start a negotiation. Parameters for the agent can be added as a dict (see example)
#   You need to specify the preference profiles for both agents. The first profile will be assigned to the first agent.
#   You need to specify a time deadline (is milliseconds (ms)) we are allowed to negotiate before we end without agreement

# Dreamteam:
# "class": "agents.ANL2022.dreamteam109_agent.dreamteam109_agent.DreamTeam109Agent",
#             "parameters": {"storage_dir": "agent_storage/DreamTeam109Agent"},
# agent44: agents.agent44.agent44.Agent44

# template_agent:
# "class": "agents.template_agent.template_agent.TemplateAgent",
#             "parameters": {"storage_dir": "agent_storage/TemplateAgent"},
settings = {
    "agents": [
        {
            "class": "agents.linear_agent.linear_agent.LinearAgent",
            "parameters": {"storage_dir": "agent_storage/LinearAgent"},
        },
        {
            "class": "agents.agent44.agent44.Agent44",
            "parameters": {"storage_dir": "agent_storage/Agent44"},
        },
    ],
    "profiles": ["domains/domain01/profileA.json", "domains/domain01/profileB.json"],
    "deadline_time_ms": 10000,
}
mode = 'session'
rounds = 10

if mode == 'session':
    session_results_summaries = {"num_offers": [], "agent_1": None, "utility_1": [], "agent_2": None, "utility_2": [], "nash_product": [], "social_welfare": [], "result": []}
    for round in range(rounds):
        print(f"Running session round {round}...")
        # run a session and obtain results in dictionaries
        session_results_trace, session_results_summary = run_session(settings)

        # plot trace to html file
        if not session_results_trace["error"]:
            plot_trace(session_results_trace, RESULTS_DIR.joinpath(f"trace_plot_round{round}.html"))

        # write results to file
        with open(RESULTS_DIR.joinpath(f"round{round}_session_results_trace.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(session_results_trace, indent=2))
        with open(RESULTS_DIR.joinpath(f"round{round}_session_results_summary.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(session_results_summary, indent=2))
        for key in session_results_summary:
            if key.startswith("agent"):
                modified_key = str(key).split("_")[0] + "_" + str(int(key.split("_")[1]) % 2 + 1)
                session_results_summaries[modified_key] = session_results_summary[key]
            elif key.startswith("utility"):
                modified_key = str(key).split("_")[0] + "_" + str(int(key.split("_")[1]) % 2 + 1)
                session_results_summaries[modified_key].append(session_results_summary[key])
            else:
                session_results_summaries[key].append(session_results_summary[key])

        session_results_summaries["avg_num_offers"] = sum(session_results_summaries["num_offers"]) / rounds
        session_results_summaries["avg_utility_1"] = sum(session_results_summaries["utility_1"]) / rounds
        session_results_summaries["avg_utility_2"] = sum(session_results_summaries["utility_2"]) / rounds
        session_results_summaries["avg_nash_product"] = sum(session_results_summaries["nash_product"]) / rounds
        session_results_summaries["avg_social_welfare"] = sum(session_results_summaries["social_welfare"]) / rounds
        session_results_summaries["num_fails"] = sum([1 if res != "agreement" else 0 for res in session_results_summaries["result"]])

        with open(RESULTS_DIR.joinpath(f"all_rounds_session_results_summary.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(session_results_summaries, indent=2))


elif mode == 'tournament':
    # run a tournament
    tournament_steps, tournament_results, tournament_results_summary = run_tournament(settings)

    with open(RESULTS_DIR.joinpath("tournament_results_summary.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(tournament_results_summary, indent=2))
