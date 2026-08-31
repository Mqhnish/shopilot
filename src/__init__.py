"""Conversational shopping agent for TechJam 2026 Track 4.

Layering (each module imports only from the ones above it):

    normalize   text canonicalisation, tokenisation, phrase keys
    catalog     the frozen 50k catalog, parsed once into flat arrays
    lexical     BM25 + exact-phrase inverted indexes
    vector      TF-IDF sparse-cosine retrieval (the vector track)
    parse       simulator utterance -> structured observations
    state       per-session constraint accumulation and override handling
    route       Buying / Browsing dual-track decision
    rank        multi-route fusion, reranking, MMR diversification
    clarify     expected-information-gain question selection
    agent       the turn loop that ties it together
"""

__version__ = "1.0.0"
