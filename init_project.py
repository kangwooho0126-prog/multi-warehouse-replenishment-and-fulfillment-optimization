from pathlib import Path

PROJECT_NAME = "multi-warehouse-replenishment-and-fulfillment-optimization"

folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "results"
]

files = {
    "README.md": "# Multi-Warehouse Replenishment and Fulfillment Optimization\n",
    "requirements.txt": "",
    "main.py": "",
    "src/__init__.py": "",
    "src/step1_build_daily_sales.py": "",
    "src/step1_5_build_project_subset.py": ""
}

def create_project():
    project_root = Path(PROJECT_NAME)
    project_root.mkdir(exist_ok=True)

    for folder in folders:
        (project_root / folder).mkdir(parents=True, exist_ok=True)

    for file_path, content in files.items():
        full_path = project_root / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        if not full_path.exists():
            full_path.write_text(content, encoding="utf-8")

    print(f"Project '{PROJECT_NAME}' has been created successfully.")
    print("\nCreated folders:")
    for folder in folders:
        print(f"  - {folder}")

    print("\nCreated files:")
    for file_path in files:
        print(f"  - {file_path}")

if __name__ == "__main__":
    create_project()