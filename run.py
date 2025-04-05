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

ALL_AGENTS = [
    "agents.ANL2022.agent007.agent007.Agent007",
    "agents.ANL2022.agent4410.agent_4410.Agent4410",
    "agents.ANL2022.agentfish.agentfish.AgentFish",
    "agents.ANL2022.BIU_agent.BIU_agent.BIU_agent",
    "agents.ANL2022.charging_boul.charging_boul.ChargingBoul",
    "agents.ANL2022.compromising_agent.compromising_agent.CompromisingAgent",
    "agents.ANL2022.dreamteam109_agent.dreamteam109_agent.DreamTeam109Agent",
    "agents.ANL2022.learning_agent.learning_agent.LearningAgent",
    "agents.ANL2022.LuckyAgent2022.LuckyAgent2022.LuckyAgent2022",
    "agents.ANL2022.micro_agent.micro_agent.micro_agent.MiCROAgent",
    "agents.ANL2022.Pinar_Agent.Pinar_Agent.Pinar_Agent",
    "agents.ANL2022.procrastin_agent.procrastin_agent.ProcrastinAgent",
    "agents.ANL2022.rg_agent.rg_agent.RGAgent",
    "agents.ANL2022.smart_agent.smart_agent.SmartAgent",
    "agents.ANL2022.super_agent.super_agent.SuperAgent",
    "agents.ANL2022.thirdagent.third_agent.ThirdAgent",
    "agents.ANL2022.tjaronchery10_agent.tjaronchery10_agent.Tjaronchery10Agent",
    "agents.agent44.agent44.Agent44",
]


def get_settings_iterator(num_domains=50):
    for dom in range(0, num_domains):
        # # everytime there is a 80% chance to skip a domain
        # if np.random.rand() < 0.8:
        #     continue
        print("=" * 50)
        print(f"Running domain {dom:02d}...")
        print("=" * 50 + "\n")
        for agent in ALL_AGENTS:
            print("=" * 50)
            print(f"Running agent {agent}...")
            print("=" * 50 + "\n")
            settings = {
                "agents": [
                    {
                        "class": agent,
                        "parameters": {"storage_dir": f"agent_storage/{agent.split('.')[-1]}"},
                    },
                    {
                        "class": "agents.group44_agent.agent44.Agent44",
                        "parameters": {"storage_dir": "self_storage/Agent44"},
                    },
                ],
                "profiles": [f"domains/domain{dom:02d}/profileA.json", f"domains/domain{dom:02d}/profileB.json"],
                "deadline_time_ms": 10000,
            }
            yield settings, dom


mode = 'session'
rounds = 1

if mode == 'session':
    session_results_summaries = {"num_offers": [], "agent_1": None, "utility_1": [], "agent_2": None, "utility_2": [], "nash_product": [], "social_welfare": [], "result": []}
    for settings, dom in get_settings_iterator():
        RESULTS_DIR_AGENT_TMP = RESULTS_DIR.joinpath(settings["agents"][0]["class"].split(".")[-1])
        if not RESULTS_DIR_AGENT_TMP.exists():
            RESULTS_DIR_AGENT_TMP.mkdir(parents=True)
        RESULTS_DIR_AGENT = RESULTS_DIR_AGENT_TMP.joinpath(f"domain{dom:02d}")
        if not RESULTS_DIR_AGENT.exists():
            RESULTS_DIR_AGENT.mkdir(parents=True)
        print(f"Running session with settings: {settings}")
        for round in range(rounds):
            print(f"Running session round {round}...")
            # run a session and obtain results in dictionaries
            session_results_trace, session_results_summary = run_session(settings)

            # plot trace to html file
            if not session_results_trace["error"]:
                plot_trace(session_results_trace, RESULTS_DIR_AGENT.joinpath(f"trace_plot_round{round}.html"))

            # write results to file
            with open(RESULTS_DIR_AGENT.joinpath(f"round{round}_session_results_trace.json"), "w", encoding="utf-8") as f:
                f.write(json.dumps(session_results_trace, indent=2))
            with open(RESULTS_DIR_AGENT.joinpath(f"round{round}_session_results_summary.json"), "w", encoding="utf-8") as f:
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

            with open(RESULTS_DIR_AGENT.joinpath(f"all_rounds_session_results_summary.json"), "w", encoding="utf-8") as f:
                f.write(json.dumps(session_results_summaries, indent=2))


elif mode == 'tournament':
    settings = {
        "agents": [
            {
                "class": "agents.ANL2022.agent007.agent007.Agent007",
                "parameters": {"storage_dir": f"agent_storage/Agent007"},
            },
            {
                "class": "agents.group44_agent.agent44.Agent44",
                "parameters": {"storage_dir": "self_storage/Agent44"},
            },
        ],
        "profiles": [f"domains/domain00/profileA.json", f"domains/domain00/profileB.json"],
        "deadline_time_ms": 10000,
    }
    # run a tournament
    tournament_steps, tournament_results, tournament_results_summary = run_tournament(settings)

    with open(RESULTS_DIR.joinpath("tournament_results_summary.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(tournament_results_summary, indent=2))
