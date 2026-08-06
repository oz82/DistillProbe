"""Probing modules. Concrete modules register here to be discoverable from the CLI."""

from .base import ProbeModule, ProbeResult
from .downstream_negation import DownstreamNegationModule
from .idiomaticity import IdiomaticityModule, IdiomaticityModuleV2
from .morphology import MorphologyModule
from .negation import NegationModule
from .polysemy import PolysemyModule
from .sts import STSModule
from .syntax import SyntaxModule, SyntaxModuleV2
from .tatoeba import TatoebaModule
from .xlingual import XLingualModule

REGISTRY: dict[str, type[ProbeModule]] = {
    "morphology": MorphologyModule,
    "idiomaticity": IdiomaticityModule,
    "idiomaticity_v2": IdiomaticityModuleV2,
    "polysemy": PolysemyModule,
    "syntax": SyntaxModule,
    "syntax_v2": SyntaxModuleV2,
    "xlingual": XLingualModule,
    "negation": NegationModule,
    "sts": STSModule,
    "downstream_negation": DownstreamNegationModule,
    "tatoeba": TatoebaModule,
}

__all__ = [
    "ProbeModule",
    "ProbeResult",
    "REGISTRY",
    "MorphologyModule",
    "IdiomaticityModule",
    "IdiomaticityModuleV2",
    "NegationModule",
    "PolysemyModule",
    "SyntaxModule",
    "STSModule",
    "SyntaxModuleV2",
    "XLingualModule",
    "DownstreamNegationModule",
    "TatoebaModule",
]
