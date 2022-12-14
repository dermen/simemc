
import logging
import copy
import time

from dials.array_family import flex
from dials.algorithms.indexing.indexer import Indexer

logger = logging.getLogger()
from simemc import utils


def index(phil, experiments, reflections, greedy=True):
    """
    :param phil: path to a stills_process phil file
    :param experiments: experiments list with crystal models
    :param reflections: strong spot reflections (observed, not necessarilly indexed)
    :return:
    """
    params = utils.stills_process_params_from_file(phil)

    st = time.time()

    logger.info("*" * 80)
    logger.info("Indexing Strong Spots")
    logger.info("*" * 80)

    params = copy.deepcopy(params)
    params.refinement.parameterisation.scan_varying = False
    if greedy:
        params.indexing.index_assignment.simple.hkl_tolerance = 0.5  # go for broke!
        params.indexing.refinement_protocol.mode = "repredict_only"  # no more refinement from dials

    idxr = Indexer.from_parameters(
        reflections,
        experiments,
        known_crystal_models=experiments.crystals(),
        params=params,
    )
    idxr.index()
    logger.info("indexed from known orientation")

    indexed = idxr.refined_reflections
    experiments = idxr.refined_experiments

    filtered_sel = flex.bool(len(indexed), True)
    for expt_id in range(len(experiments)):
        for idx in set(
            indexed["miller_index"].select(indexed["id"] == expt_id)
        ):
            sel = (indexed["miller_index"] == idx) & (indexed["id"] == expt_id)
            if sel.count(True) > 1:
                filtered_sel = filtered_sel & ~sel
    filtered = indexed.select(filtered_sel)
    logger.info(
        "Filtered duplicate reflections, %d out of %d remaining",
        len(filtered),
        len(indexed),
    )
    print(
        "Filtered duplicate reflections, %d out of %d remaining"
        % (len(filtered), len(indexed))
    )
    indexed = filtered

    logger.info("")
    logger.info("Time Taken = %f seconds", time.time() - st)
    return experiments, indexed
