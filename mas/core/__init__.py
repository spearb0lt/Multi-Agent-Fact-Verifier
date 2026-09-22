"""Shared infrastructure the agent layer stands on.

Nothing in here knows what an agent is. It provides model access, embeddings,
web access, storage and cost accounting, and the `mas.kernel` layer composes
those into an agent runtime.
"""
