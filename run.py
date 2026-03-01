#!/usr/bin/env python3
"""
Entry point — run this to start the agent.
Usage:  python run.py
"""

from main import TradingOrchestrator

if __name__ == "__main__":
    agent = TradingOrchestrator()
    agent.run()
