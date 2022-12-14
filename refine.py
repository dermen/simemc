import logging
logger = logging.getLogger()
import time

from simemc import utils
from dials.algorithms.refinement import RefinerFactory


def refine(phil, experiments, centroids):
    params = utils.stills_process_params_from_file(phil)

    st = time.time()

    logger.info("*" * 80)
    logger.info("Refining Model")
    logger.info("*" * 80)

    refiner = RefinerFactory.from_parameters_data_experiments(
        params, centroids, experiments
    )

    refiner.run()
    experiments = refiner.get_experiments()
    predicted = refiner.predict_for_indexed()
    centroids["xyzcal.mm"] = predicted["xyzcal.mm"]
    centroids["entering"] = predicted["entering"]
    centroids = centroids.select(refiner.selection_used_for_refinement())

    # Re-estimate mosaic estimates
    from dials.algorithms.indexing.nave_parameters import NaveParameters

    nv = NaveParameters(
        params=params,
        experiments=experiments,
        reflections=centroids,
        refinery=refiner,
        graph_verbose=False,
    )
    nv()
    acceptance_flags_nv = nv.nv_acceptance_flags
    centroids = centroids.select(acceptance_flags_nv)

    return experiments, centroids
