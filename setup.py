from setuptools import setup, find_packages

setup(
    name="agentarmor",
    version="0.1.0",
    description="Securing LLM Agents via Structured Graph Abstraction",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "networkx>=3.0",
        "pydantic>=2.0",
        "openai>=1.0",
    ],
)
