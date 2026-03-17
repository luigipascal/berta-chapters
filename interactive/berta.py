#!/usr/bin/env python3
"""
Berta Interactive Learning Hub - v1.1

The main entry point for the Berta Chapters interactive experience.
Provides path selection, progress tracking, chapter navigation,
and a guided learning journey through the AI curriculum.

Usage:
    python interactive/berta.py              # Launch the interactive hub
    python interactive/berta.py paths        # Show learning paths
    python interactive/berta.py status       # Show your progress
    python interactive/berta.py chapter 1    # Open chapter 1 info
    python interactive/berta.py quiz         # Quick knowledge check
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, IntPrompt, Confirm
    from rich.markdown import Markdown
    from rich.tree import Tree
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.columns import Columns
    from rich.text import Text
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

PROGRESS_FILE = Path(__file__).parent.parent / ".berta_progress.json"

CHAPTERS = {
    1: {"title": "Python Fundamentals for AI", "track": "Foundation", "hours": 8, "prereqs": []},
    2: {"title": "Data Structures & Algorithms", "track": "Foundation", "hours": 6, "prereqs": [1]},
    3: {"title": "Linear Algebra & Calculus", "track": "Foundation", "hours": 10, "prereqs": [1]},
    4: {"title": "Probability & Statistics", "track": "Foundation", "hours": 8, "prereqs": [1]},
    5: {"title": "Software Design & Best Practices", "track": "Foundation", "hours": 6, "prereqs": [1, 2]},
    6: {"title": "Introduction to Machine Learning", "track": "Practitioner", "hours": 8, "prereqs": [1, 2, 3, 4]},
    7: {"title": "Supervised Learning", "track": "Practitioner", "hours": 10, "prereqs": [1, 2, 3, 4, 5, 6]},
    8: {"title": "Unsupervised Learning", "track": "Practitioner", "hours": 8, "prereqs": [1, 2, 3, 4, 5, 6]},
    9: {"title": "Deep Learning Fundamentals", "track": "Practitioner", "hours": 12, "prereqs": [1, 3, 6]},
    10: {"title": "Natural Language Processing", "track": "Practitioner", "hours": 10, "prereqs": [1, 2, 3, 4, 5, 6, 9]},
    11: {"title": "Large Language Models & Transformers", "track": "Practitioner", "hours": 10, "prereqs": [1, 9, 10]},
    12: {"title": "Prompt Engineering", "track": "Practitioner", "hours": 6, "prereqs": [1, 11]},
    13: {"title": "Retrieval-Augmented Generation", "track": "Practitioner", "hours": 8, "prereqs": [1, 10, 11, 12]},
    14: {"title": "Fine-tuning & Adaptation", "track": "Practitioner", "hours": 8, "prereqs": [1, 9, 11]},
    15: {"title": "MLOps & Deployment", "track": "Practitioner", "hours": 8, "prereqs": [1, 2, 3, 4, 5, 6]},
    16: {"title": "Multi-Agent Systems", "track": "Advanced", "hours": 10, "prereqs": [1, 9, 11, 20]},
    17: {"title": "Advanced RAG & Knowledge Systems", "track": "Advanced", "hours": 10, "prereqs": [1, 13]},
    18: {"title": "Reinforcement Learning", "track": "Advanced", "hours": 12, "prereqs": [1, 3, 6, 9]},
    19: {"title": "Model Optimization & Inference", "track": "Advanced", "hours": 8, "prereqs": [1, 9, 15]},
    20: {"title": "Production AI Systems", "track": "Advanced", "hours": 10, "prereqs": [1, 2, 3, 4, 5, 6, 15]},
    21: {"title": "AI for Finance", "track": "Advanced", "hours": 10, "prereqs": [1, 3, 4, 6, 7]},
    22: {"title": "AI Safety & Alignment", "track": "Advanced", "hours": 8, "prereqs": [1, 9, 11]},
    23: {"title": "Building AI Products", "track": "Advanced", "hours": 8, "prereqs": [1, 2, 3, 4, 5, 6, 15, 20]},
    24: {"title": "Research & Cutting-Edge", "track": "Advanced", "hours": 8, "prereqs": [1, 9, 11, 18]},
    25: {"title": "AI Governance & Ethics", "track": "Advanced", "hours": 6, "prereqs": [1, 20, 22]},
}

LEARNING_PATHS = {
    "A": {
        "name": "Complete AI Engineer",
        "chapters": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20],
        "hours": 110,
        "description": "Full-stack AI engineering, systems thinking, deployment",
        "best_for": "Career changers, aspiring ML engineers",
    },
    "B": {
        "name": "Machine Learning Specialist",
        "chapters": [1, 2, 3, 4, 6, 7, 8, 9, 15, 19, 20],
        "hours": 100,
        "description": "Advanced ML techniques, optimization, production systems",
        "best_for": "Data scientists, ML engineers, researchers",
    },
    "C": {
        "name": "LLM & NLP Expert",
        "chapters": [1, 5, 10, 11, 12, 13, 14, 17, 20, 23],
        "hours": 90,
        "description": "LLM expertise, RAG systems, fine-tuning, production NLP",
        "best_for": "NLP engineers, LLM application builders",
    },
    "D": {
        "name": "AI for Finance",
        "chapters": [1, 3, 4, 6, 7, 21, 19, 20, 23],
        "hours": 85,
        "description": "Financial ML models, trading systems, risk analysis",
        "best_for": "Finance professionals, quants, fintech engineers",
    },
    "E": {
        "name": "Quick Start: AI Fundamentals",
        "chapters": [1, 5, 6, 9, 11, 23],
        "hours": 48,
        "description": "Core AI concepts, ability to build simple AI applications",
        "best_for": "Quick learners, career explorers, busy professionals",
    },
    "F": {
        "name": "Executive / Manager",
        "chapters": [5, 6, 20, 22, 23, 25],
        "hours": 38,
        "description": "Understanding AI capabilities, limitations, strategic thinking",
        "best_for": "Executives, managers, founders",
    },
}

AVAILABLE_CHAPTERS = {1, 2, 3, 4, 5, 6, 7}

QUIZ_QUESTIONS = [
    {
        "question": "What data type in Python is ordered, mutable, and allows duplicates?",
        "options": ["Set", "Dictionary", "List", "Tuple"],
        "answer": 3,
        "chapter": 1,
        "explanation": "Lists are ordered, mutable, and allow duplicate elements. Sets are unordered and don't allow duplicates."
    },
    {
        "question": "What is the time complexity of binary search on a sorted array?",
        "options": ["O(n)", "O(log n)", "O(n log n)", "O(1)"],
        "answer": 2,
        "chapter": 2,
        "explanation": "Binary search divides the search space in half each step, giving O(log n) complexity."
    },
    {
        "question": "In machine learning, what is overfitting?",
        "options": [
            "Model performs poorly on both training and test data",
            "Model performs well on training data but poorly on new data",
            "Model is too simple to capture patterns",
            "Model takes too long to train",
        ],
        "answer": 2,
        "chapter": 6,
        "explanation": "Overfitting means the model memorized training data patterns (including noise) instead of learning generalizable patterns."
    },
    {
        "question": "What does the 'T' in Transformer stand for in the context of GPT?",
        "options": ["Transfer", "Transformer", "Training", "Temporal"],
        "answer": 2,
        "chapter": 11,
        "explanation": "GPT = Generative Pre-trained Transformer. The Transformer is the architecture that uses self-attention."
    },
    {
        "question": "What is RAG in the context of LLMs?",
        "options": [
            "Random Access Generation",
            "Retrieval-Augmented Generation",
            "Recursive Algorithm Gateway",
            "Reinforced Agent Graph",
        ],
        "answer": 2,
        "chapter": 13,
        "explanation": "RAG combines retrieval of relevant documents with LLM generation for grounded, knowledge-based responses."
    },
    {
        "question": "Which technique reduces model size by representing weights with fewer bits?",
        "options": ["Pruning", "Distillation", "Quantization", "Dropout"],
        "answer": 3,
        "chapter": 19,
        "explanation": "Quantization represents model weights with fewer bits (e.g., INT8 instead of FP32) to reduce model size and speed up inference."
    },
    {
        "question": "What is the primary purpose of a vector database in AI systems?",
        "options": [
            "Store relational data",
            "Run SQL queries",
            "Enable similarity search over embeddings",
            "Train neural networks",
        ],
        "answer": 3,
        "chapter": 13,
        "explanation": "Vector databases store embeddings and enable efficient similarity search, powering RAG and semantic search applications."
    },
    {
        "question": "In reinforcement learning, what is the explore-exploit dilemma?",
        "options": [
            "Choosing between CPU and GPU",
            "Balancing trying new actions vs using known good actions",
            "Deciding model architecture size",
            "Selecting training vs test data",
        ],
        "answer": 2,
        "chapter": 18,
        "explanation": "The agent must balance exploring new strategies (which might be better) versus exploiting what already works well."
    },
]


def load_progress():
    default = {
        "learner_name": None,
        "selected_path": None,
        "chapters_completed": [],
        "chapters_in_progress": [],
        "quiz_scores": [],
        "started_at": None,
        "last_active": None,
    }
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return default
    return default


def save_progress(progress):
    progress["last_active"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def plain_welcome(progress):
    print("\n" + "=" * 60)
    print("  BERTA CHAPTERS - Interactive AI Learning Hub")
    print("=" * 60)
    name = progress.get("learner_name", "Learner")
    if name:
        print(f"\n  Welcome back, {name}!")
    else:
        print("\n  Welcome to Berta Chapters!")
    print("\n  Learn AI from fundamentals to mastery through")
    print("  interactive, executable chapters.")
    print("\n  Free. Open-source. Community-driven.")
    print("-" * 60)


def plain_menu():
    print("\n  What would you like to do?\n")
    print("  [1] Explore Learning Paths")
    print("  [2] View All Chapters")
    print("  [3] Check My Progress")
    print("  [4] Take a Quick Quiz")
    print("  [5] Chapter Deep-Dive")
    print("  [6] Set Up My Profile")
    print("  [7] Skill Assessment")
    print("  [0] Exit\n")
    try:
        return input("  Your choice: ").strip()
    except (EOFError, KeyboardInterrupt):
        return "0"


def run_plain_mode():
    progress = load_progress()
    plain_welcome(progress)
    while True:
        choice = plain_menu()
        if choice == "0":
            print("\n  Happy learning! See you next time.\n")
            break
        elif choice == "1":
            print("\n  LEARNING PATHS")
            print("  " + "-" * 40)
            for key, path in LEARNING_PATHS.items():
                ch_str = " -> ".join(str(c) for c in path["chapters"])
                print(f"\n  Path {key}: {path['name']}")
                print(f"    Hours: {path['hours']}  |  Best for: {path['best_for']}")
                print(f"    Chapters: {ch_str}")
                print(f"    {path['description']}")
        elif choice == "2":
            print("\n  ALL CHAPTERS")
            print("  " + "-" * 40)
            for num, ch in CHAPTERS.items():
                status = "Done" if num in progress.get("chapters_completed", []) else "..."
                print(f"  [{status:>4}] Ch {num:>2}: {ch['title']} ({ch['hours']}h) [{ch['track']}]")
        elif choice == "3":
            completed = progress.get("chapters_completed", [])
            total_hours = sum(CHAPTERS[c]["hours"] for c in completed if c in CHAPTERS)
            print(f"\n  Completed: {len(completed)} / {len(CHAPTERS)} chapters")
            print(f"  Hours invested: {total_hours}h")
        elif choice == "4":
            import random
            q = random.choice(QUIZ_QUESTIONS)
            print(f"\n  QUIZ (Chapter {q['chapter']})")
            print(f"  {q['question']}\n")
            for i, opt in enumerate(q["options"], 1):
                print(f"    [{i}] {opt}")
            try:
                ans = input("\n  Your answer: ").strip()
                if ans.isdigit() and int(ans) == q["answer"]:
                    print(f"  Correct! {q['explanation']}")
                else:
                    print(f"  Not quite. {q['explanation']}")
            except (EOFError, KeyboardInterrupt):
                pass
        elif choice == "6":
            try:
                name = input("  Your name: ").strip()
                if name:
                    progress["learner_name"] = name
                    save_progress(progress)
                    print(f"  Profile updated! Welcome, {name}.")
            except (EOFError, KeyboardInterrupt):
                pass
        else:
            print("  Coming soon!")


class BertaHub:
    """Rich-powered interactive learning hub."""

    def __init__(self):
        self.console = Console()
        self.progress = load_progress()

    def welcome_banner(self):
        banner = Text()
        banner.append("B E R T A", style="bold cyan")
        banner.append("  ", style="")
        banner.append("C H A P T E R S", style="bold white")

        subtitle = Text()
        subtitle.append("Interactive AI Learning Hub", style="italic dim")

        name = self.progress.get("learner_name")
        greeting = f"Welcome back, {name}!" if name else "Welcome, Learner!"

        content = Text.assemble(
            ("", ""),
            banner, "\n\n",
            subtitle, "\n\n",
            (greeting, "bold green"), "\n",
            ("Learn AI from fundamentals to mastery", "dim"),
            ("\nFree. Open-source. Community-driven.", "dim"),
        )

        panel = Panel(
            content,
            border_style="cyan",
            box=box.DOUBLE_EDGE,
            padding=(1, 4),
        )
        self.console.print(panel)
        self._show_quick_stats()

    def _show_quick_stats(self):
        completed = self.progress.get("chapters_completed", [])
        in_progress = self.progress.get("chapters_in_progress", [])
        total_hours = sum(CHAPTERS[c]["hours"] for c in completed if c in CHAPTERS)
        pct = (len(completed) / len(CHAPTERS)) * 100

        stats = Table(show_header=False, box=None, padding=(0, 2))
        stats.add_column(style="bold cyan")
        stats.add_column(style="bold white")
        stats.add_column(style="bold cyan")
        stats.add_column(style="bold white")

        stats.add_row(
            "Completed", f"{len(completed)}/{len(CHAPTERS)}",
            "In Progress", str(len(in_progress)),
        )
        stats.add_row(
            "Hours", f"{total_hours}h",
            "Progress", f"{pct:.0f}%",
        )
        self.console.print(Panel(stats, title="Your Journey", border_style="dim"))

    def main_menu(self):
        menu_items = [
            ("[1]", "Explore Learning Paths", "Find your ideal journey through AI"),
            ("[2]", "Browse All Chapters", "See every chapter across all tracks"),
            ("[3]", "My Progress Dashboard", "Track your learning journey"),
            ("[4]", "Quick Knowledge Quiz", "Test yourself with AI questions"),
            ("[5]", "Chapter Deep-Dive", "Explore a specific chapter in detail"),
            ("[6]", "Skill Assessment", "Discover where to start based on your skills"),
            ("[7]", "Set Up Profile", "Personalize your learning experience"),
            ("[0]", "Exit", "Save progress and exit"),
        ]

        table = Table(
            show_header=False,
            box=box.SIMPLE_HEAVY,
            border_style="cyan",
            padding=(0, 2),
            title="What would you like to do?",
            title_style="bold",
        )
        table.add_column(style="bold yellow", width=5)
        table.add_column(style="bold white", width=28)
        table.add_column(style="dim")

        for key, title, desc in menu_items:
            table.add_row(key, title, desc)

        self.console.print()
        self.console.print(table)
        self.console.print()

        return Prompt.ask("Your choice", choices=["0", "1", "2", "3", "4", "5", "6", "7"], default="1")

    def show_learning_paths(self, interactive=True):
        self.console.print("\n[bold cyan]Learning Paths[/bold cyan]\n")

        for key, path in LEARNING_PATHS.items():
            completed = self.progress.get("chapters_completed", [])
            done = sum(1 for c in path["chapters"] if c in completed)
            total = len(path["chapters"])
            pct = (done / total) * 100 if total > 0 else 0

            ch_display = []
            for c in path["chapters"]:
                if c in completed:
                    ch_display.append(f"[green][{c}][/green]")
                else:
                    ch_display.append(f"[dim]{c}[/dim]")

            content = Text.assemble(
                (f"{path['description']}\n", ""),
                (f"Best for: ", "dim"),
                (f"{path['best_for']}\n", "italic"),
                (f"Time: {path['hours']} hours  |  ", "dim"),
                (f"Progress: {done}/{total} ({pct:.0f}%)\n", "bold"),
            )

            panel = Panel(
                content,
                title=f"[bold]Path {key}: {path['name']}[/bold]",
                subtitle=" -> ".join(ch_display),
                border_style="green" if pct == 100 else "cyan" if pct > 0 else "dim",
            )
            self.console.print(panel)

        if not interactive:
            return

        self.console.print()
        choice = Prompt.ask(
            "Select a path to follow (or 'back')",
            choices=list(LEARNING_PATHS.keys()) + ["back"],
            default="back",
        )
        if choice != "back" and choice in LEARNING_PATHS:
            self.progress["selected_path"] = choice
            save_progress(self.progress)
            path = LEARNING_PATHS[choice]
            self.console.print(
                f"\n[bold green]Path {choice}: {path['name']} selected![/bold green]"
            )
            self.console.print(f"Start with Chapter {path['chapters'][0]}: {CHAPTERS[path['chapters'][0]]['title']}")

    def browse_chapters(self):
        self.console.print("\n[bold cyan]All Chapters[/bold cyan]\n")
        completed = self.progress.get("chapters_completed", [])
        in_prog = self.progress.get("chapters_in_progress", [])

        for track in ["Foundation", "Practitioner", "Advanced"]:
            track_colors = {"Foundation": "blue", "Practitioner": "magenta", "Advanced": "green"}
            color = track_colors[track]
            table = Table(
                title=f"{track} Track",
                title_style=f"bold {color}",
                box=box.ROUNDED,
                border_style=color,
            )
            table.add_column("#", style="bold", width=4, justify="right")
            table.add_column("Chapter", style="white", min_width=35)
            table.add_column("Hours", justify="center", width=6)
            table.add_column("Content", justify="center", width=12)
            table.add_column("Status", justify="center", width=14)
            table.add_column("Prerequisites", style="dim", width=20)

            for num, ch in CHAPTERS.items():
                if ch["track"] != track:
                    continue
                if num in completed:
                    status = "[bold green]Completed[/bold green]"
                elif num in in_prog:
                    status = "[bold yellow]In Progress[/bold yellow]"
                else:
                    status = "[dim]Not Started[/dim]"

                available = "[green]Available[/green]" if num in AVAILABLE_CHAPTERS else "[dim]Planned[/dim]"
                prereqs = ", ".join(str(p) for p in ch["prereqs"]) if ch["prereqs"] else "None"
                table.add_row(str(num), ch["title"], str(ch["hours"]), available, status, prereqs)

            self.console.print(table)
            self.console.print()

    def show_progress(self):
        completed = self.progress.get("chapters_completed", [])
        in_prog = self.progress.get("chapters_in_progress", [])
        total_hours = sum(CHAPTERS[c]["hours"] for c in completed if c in CHAPTERS)
        prog_hours = sum(CHAPTERS[c]["hours"] for c in in_prog if c in CHAPTERS)

        self.console.print("\n[bold cyan]Progress Dashboard[/bold cyan]\n")

        overview = Table(show_header=False, box=box.SIMPLE, padding=(0, 3))
        overview.add_column(style="bold cyan", width=22)
        overview.add_column(style="bold white", width=15)
        overview.add_row("Chapters Completed", f"{len(completed)} / {len(CHAPTERS)}")
        overview.add_row("Hours Completed", f"{total_hours}h")
        overview.add_row("Hours In Progress", f"{prog_hours}h")
        overview.add_row("Quiz Score Avg",
                         f"{sum(self.progress.get('quiz_scores', [0])) / max(len(self.progress.get('quiz_scores', [1])), 1):.0f}%")
        selected = self.progress.get("selected_path")
        if selected:
            overview.add_row("Current Path", f"Path {selected}: {LEARNING_PATHS[selected]['name']}")
        self.console.print(Panel(overview, title="Overview", border_style="cyan"))

        if selected and selected in LEARNING_PATHS:
            path = LEARNING_PATHS[selected]
            self.console.print(f"\n[bold]Path {selected}: {path['name']} Progress[/bold]\n")
            with Progress(
                TextColumn("[bold]{task.description}"),
                BarColumn(bar_width=40),
                TextColumn("{task.percentage:.0f}%"),
                console=self.console,
            ) as prog_bar:
                done = sum(1 for c in path["chapters"] if c in completed)
                task = prog_bar.add_task(path["name"], total=len(path["chapters"]))
                prog_bar.update(task, completed=done)
            self.console.print()

        if completed:
            tree = Tree("[bold green]Completed Chapters[/bold green]")
            for c in sorted(completed):
                if c in CHAPTERS:
                    tree.add(f"[green]Ch {c}: {CHAPTERS[c]['title']}[/green]")
            self.console.print(tree)

        if in_prog:
            tree = Tree("[bold yellow]In Progress[/bold yellow]")
            for c in sorted(in_prog):
                if c in CHAPTERS:
                    tree.add(f"[yellow]Ch {c}: {CHAPTERS[c]['title']}[/yellow]")
            self.console.print(tree)

        self.console.print()
        action = Prompt.ask(
            "Action",
            choices=["mark_complete", "start_chapter", "back"],
            default="back",
        )
        if action == "mark_complete" and in_prog:
            ch = IntPrompt.ask("Which chapter number?")
            if ch in in_prog:
                self.progress["chapters_in_progress"].remove(ch)
                self.progress["chapters_completed"].append(ch)
                save_progress(self.progress)
                self.console.print(f"[bold green]Chapter {ch} marked complete![/bold green]")
        elif action == "start_chapter":
            ch = IntPrompt.ask("Which chapter number?")
            if ch in CHAPTERS and ch not in completed and ch not in in_prog:
                self.progress["chapters_in_progress"].append(ch)
                save_progress(self.progress)
                self.console.print(f"[bold yellow]Chapter {ch} started! Good luck![/bold yellow]")

    def quick_quiz(self):
        import random
        self.console.print("\n[bold cyan]Quick Knowledge Quiz[/bold cyan]\n")

        questions = random.sample(QUIZ_QUESTIONS, min(5, len(QUIZ_QUESTIONS)))
        score = 0

        for i, q in enumerate(questions, 1):
            self.console.print(f"[bold]Question {i}/{len(questions)}[/bold] [dim](Chapter {q['chapter']})[/dim]")
            self.console.print(f"[white]{q['question']}[/white]\n")

            for j, opt in enumerate(q["options"], 1):
                self.console.print(f"  [{j}] {opt}")

            try:
                answer = IntPrompt.ask("\nYour answer", choices=["1", "2", "3", "4"])
            except (EOFError, KeyboardInterrupt):
                break

            if answer == q["answer"]:
                score += 1
                self.console.print(f"[bold green]Correct![/bold green] {q['explanation']}\n")
            else:
                self.console.print(
                    f"[bold red]Not quite.[/bold red] The answer is [{q['answer']}]. {q['explanation']}\n"
                )

        pct = (score / len(questions)) * 100 if questions else 0
        self.progress.setdefault("quiz_scores", []).append(pct)
        save_progress(self.progress)

        color = "green" if pct >= 80 else "yellow" if pct >= 50 else "red"
        self.console.print(
            Panel(
                f"[bold {color}]{score}/{len(questions)} correct ({pct:.0f}%)[/bold {color}]",
                title="Quiz Results",
                border_style=color,
            )
        )

    def chapter_deep_dive(self):
        self.console.print("\n[bold cyan]Chapter Deep-Dive[/bold cyan]\n")
        ch_num = IntPrompt.ask("Enter chapter number (1-25)")

        if ch_num not in CHAPTERS:
            self.console.print("[red]Chapter not found.[/red]")
            return

        ch = CHAPTERS[ch_num]
        completed = self.progress.get("chapters_completed", [])

        prereq_status = []
        for p in ch["prereqs"]:
            if p in completed:
                prereq_status.append(f"[green]Ch {p}: {CHAPTERS[p]['title']}[/green]")
            else:
                prereq_status.append(f"[red]Ch {p}: {CHAPTERS[p]['title']} (not completed)[/red]")

        ready = all(p in completed for p in ch["prereqs"]) or not ch["prereqs"]

        content = Text.assemble(
            ("Track: ", "dim"), (ch["track"], "bold"), ("\n", ""),
            ("Time: ", "dim"), (f"{ch['hours']} hours", "bold"), ("\n", ""),
            ("Status: ", "dim"),
            ("[green]Completed[/green]" if ch_num in completed else "[yellow]Ready[/yellow]" if ready else "[red]Prerequisites needed[/red]", ""),
            ("\n\n", ""),
            ("Prerequisites:\n", "bold"),
        )

        panel_content = str(content)
        if prereq_status:
            panel_content = (
                f"Track: {ch['track']}  |  Time: {ch['hours']} hours\n"
                f"Status: {'Completed' if ch_num in completed else 'Ready' if ready else 'Prerequisites needed'}\n\n"
                f"Prerequisites:\n" + "\n".join(f"  {p}" for p in prereq_status)
            )
        else:
            panel_content = (
                f"Track: {ch['track']}  |  Time: {ch['hours']} hours\n"
                f"Status: {'Completed' if ch_num in completed else 'Ready' if ready else 'Prerequisites needed'}\n\n"
                f"Prerequisites: None"
            )

        self.console.print(Panel(
            panel_content,
            title=f"[bold]Chapter {ch_num}: {ch['title']}[/bold]",
            border_style="cyan",
        ))

        if ch_num in AVAILABLE_CHAPTERS:
            chapter_dir = Path(__file__).parent.parent / "chapters" / f"chapter-{ch_num:02d}-*"
            import glob
            matches = glob.glob(str(chapter_dir))
            if matches:
                self.console.print(f"\n[green]Chapter content available at:[/green] {matches[0]}")
                self.console.print("[dim]Contains: 3 notebooks, scripts, exercises with solutions[/dim]")
        else:
            self.console.print(f"\n[yellow]Chapter content coming soon.[/yellow] Check the roadmap for release dates.")

    def skill_assessment(self):
        self.console.print("\n[bold cyan]Skill Assessment[/bold cyan]")
        self.console.print("[dim]Answer a few questions to find your starting point.[/dim]\n")

        skills = {"python": 0, "math": 0, "ml": 0, "dl": 0, "llm": 0}

        questions = [
            ("How comfortable are you with Python?", "python",
             ["Never used it", "Written some scripts", "Comfortable with OOP", "Advanced (decorators, generators, async)"]),
            ("How's your math background?", "math",
             ["High school algebra", "Some college math", "Linear algebra + calculus", "Advanced (proofs, optimization)"]),
            ("Machine Learning experience?", "ml",
             ["What is ML?", "Understand concepts, no coding", "Built models with sklearn", "Production ML systems"]),
            ("Deep Learning experience?", "dl",
             ["Never tried", "Completed a tutorial", "Built models with PyTorch/TF", "Research-level"]),
            ("LLM / NLP experience?", "llm",
             ["Just a user (ChatGPT)", "Basic prompt engineering", "Built RAG / fine-tuned models", "Developed LLM-based products"]),
        ]

        for question, skill, options in questions:
            self.console.print(f"[bold]{question}[/bold]")
            for i, opt in enumerate(options):
                self.console.print(f"  [{i + 1}] {opt}")
            try:
                answer = IntPrompt.ask("Your answer", choices=["1", "2", "3", "4"])
                skills[skill] = answer
            except (EOFError, KeyboardInterrupt):
                return

        total = sum(skills.values())

        if total <= 7:
            rec_path, start_ch = "E", 1
            level = "Beginner"
        elif total <= 12:
            rec_path, start_ch = "A", 5
            level = "Intermediate"
        elif total <= 17:
            rec_path, start_ch = "B" if skills["ml"] >= 3 else "C", 6
            level = "Advanced Beginner"
        else:
            rec_path, start_ch = "C" if skills["llm"] >= 3 else "B", 11
            level = "Advanced"

        path = LEARNING_PATHS[rec_path]

        self.console.print(Panel(
            f"[bold]Level: {level}[/bold]\n\n"
            f"Recommended Path: [cyan]Path {rec_path}: {path['name']}[/cyan]\n"
            f"Start with: [green]Chapter {start_ch}: {CHAPTERS[start_ch]['title']}[/green]\n"
            f"Estimated time: [yellow]{path['hours']} hours[/yellow]\n\n"
            f"[dim]{path['description']}[/dim]",
            title="Your Recommendation",
            border_style="green",
        ))

        if Confirm.ask("Follow this path?"):
            self.progress["selected_path"] = rec_path
            if start_ch not in self.progress.get("chapters_in_progress", []):
                self.progress.setdefault("chapters_in_progress", []).append(start_ch)
            save_progress(self.progress)
            self.console.print(f"[bold green]Path set! Chapter {start_ch} is now in progress.[/bold green]")

    def setup_profile(self):
        self.console.print("\n[bold cyan]Profile Setup[/bold cyan]\n")

        name = Prompt.ask("Your name", default=self.progress.get("learner_name", ""))
        if name:
            self.progress["learner_name"] = name

        if not self.progress.get("started_at"):
            self.progress["started_at"] = datetime.now().isoformat()

        save_progress(self.progress)
        self.console.print(f"\n[bold green]Profile saved! Welcome, {name}.[/bold green]")

    def run(self):
        self.welcome_banner()

        while True:
            try:
                choice = self.main_menu()
            except (EOFError, KeyboardInterrupt):
                break

            if choice == "0":
                save_progress(self.progress)
                self.console.print(
                    Panel(
                        "[bold]Happy learning! See you next time.[/bold]\n[dim]Your progress has been saved.[/dim]",
                        border_style="cyan",
                    )
                )
                break
            elif choice == "1":
                self.show_learning_paths()
            elif choice == "2":
                self.browse_chapters()
            elif choice == "3":
                self.show_progress()
            elif choice == "4":
                self.quick_quiz()
            elif choice == "5":
                self.chapter_deep_dive()
            elif choice == "6":
                self.skill_assessment()
            elif choice == "7":
                self.setup_profile()


def main():
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "paths":
            try:
                # In CI (e.g. GitHub Actions), use plain output to avoid Rich/non-TTY issues
                in_ci = os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CI") == "true"
                if HAS_RICH and not in_ci:
                    hub = BertaHub()
                    hub.show_learning_paths(interactive=False)
                else:
                    for key, path in LEARNING_PATHS.items():
                        print(f"\nPath {key}: {path['name']} ({path['hours']}h)")
                        print(f"  {path['description']}")
                        print(f"  Chapters: {' -> '.join(str(c) for c in path['chapters'])}")
            except Exception as e:
                print(f"paths: {e}", file=sys.stderr)
                sys.exit(1)
            sys.exit(0)
        elif cmd == "status":
            try:
                progress = load_progress()
                completed = progress.get("chapters_completed", [])
                print(f"Completed: {len(completed)}/{len(CHAPTERS)} chapters")
            except Exception as e:
                print(f"status: {e}", file=sys.stderr)
                sys.exit(1)
            sys.exit(0)
        elif cmd == "quiz":
            if HAS_RICH:
                hub = BertaHub()
                hub.quick_quiz()
            else:
                run_plain_mode()
            return
        elif cmd == "chapter" and len(sys.argv) > 2:
            ch_num = int(sys.argv[2])
            if ch_num in CHAPTERS:
                ch = CHAPTERS[ch_num]
                print(f"\nChapter {ch_num}: {ch['title']}")
                print(f"Track: {ch['track']}  |  Hours: {ch['hours']}")
                prereqs = ", ".join(str(p) for p in ch["prereqs"]) or "None"
                print(f"Prerequisites: {prereqs}")
            return

    if HAS_RICH:
        hub = BertaHub()
        hub.run()
    else:
        print("[Note: Install 'rich' for the best experience: pip install rich]")
        run_plain_mode()


if __name__ == "__main__":
    main()
