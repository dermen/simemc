
import time
import logging

from dials.array_family import flex
from dxtbx.model import ExperimentList
from dials.algorithms.integration.integrator import create_integrator
from dials.algorithms.profile_model.factory import ProfileModelFactory


from simemc import utils

logger = logging.getLogger()


def process_reference(reference):
    """Load the reference spots."""
    assert "miller_index" in reference
    assert "id" in reference
    mask = reference.get_flags(reference.flags.indexed)
    rubbish = reference.select(~mask)
    if mask.count(False) > 0:
        reference.del_selected(~mask)
    if len(reference) == 0:
        raise RuntimeError(
            """
    Invalid input for reference reflections.
    Expected > %d indexed spots, got %d
  """
            % (0, len(reference))
        )
    mask = reference["miller_index"] == (0, 0, 0)
    if mask.count(True) > 0:
        rubbish.extend(reference.select(mask))
        reference.del_selected(mask)
    mask = reference["id"] < 0
    if mask.count(True) > 0:
        raise RuntimeError(
            """
    Invalid input for reference reflections.
    %d reference spots have an invalid experiment id
  """
            % mask.count(True)
        )
    return reference, rubbish


def integrate(phil, experiments, indexed, mask=None, sig_b_cut=False):

    st = time.time()

    params = utils.stills_process_params_from_file(phil)
    if not sig_b_cut:
        params.profile.gaussian_rs.parameters.sigma_b_cutoff = None
    if mask is not None:
        params.integration.lookup.mask = mask
    logger.info("*" * 80)
    logger.info("Integrating Reflections")
    logger.info("*" * 80)

    indexed, _ = process_reference(indexed)

    if params.integration.integration_only_overrides.trusted_range:
        for detector in experiments.detectors():
            for panel in detector:
                panel.set_trusted_range(
                    params.integration.integration_only_overrides.trusted_range
                )

    if params.dispatch.coset:
        raise NotImplementedError("dispatch coset not implemented")

    # Get the integrator from the input parameters
    logger.info("Configuring integrator from input parameters")

    # Compute the profile model
    # Predict the reflections
    # Match the predictions with the reference
    # Create the integrator
    experiments = ProfileModelFactory.create(params, experiments, indexed)
    new_experiments = ExperimentList()
    new_reflections = flex.reflection_table()
    for expt_id, expt in enumerate(experiments):
        if (
            params.profile.gaussian_rs.parameters.sigma_b_cutoff is None
            or expt.profile.sigma_b()
            < params.profile.gaussian_rs.parameters.sigma_b_cutoff
        ):
            refls = indexed.select(indexed["id"] == expt_id)
            refls["id"] = flex.int(len(refls), len(new_experiments))
            # refls.reset_ids()
            del refls.experiment_identifiers()[expt_id]
            refls.experiment_identifiers()[len(new_experiments)] = expt.identifier
            new_reflections.extend(refls)
            new_experiments.append(expt)
        else:
            logger.info(
                "Rejected expt %d with sigma_b %f"
                % (expt_id, expt.profile.sigma_b())
            )
    experiments = new_experiments
    indexed = new_reflections
    if len(experiments) == 0:
        raise RuntimeError("No experiments after filtering by sigma_b")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Predicting reflections")
    logger.info("")
    predicted = flex.reflection_table.from_predictions_multi(
        experiments,
        dmin=params.prediction.d_min,
        dmax=params.prediction.d_max,
        margin=params.prediction.margin,
        force_static=params.prediction.force_static,
    )
    predicted.match_with_reference(indexed)
    logger.info("")
    integrator = create_integrator(params, experiments, predicted)

    # Integrate the reflections
    integrated = integrator.integrate()

    # correct integrated intensities for absorption correction, if necessary
    for abs_params in params.integration.absorption_correction:
        if abs_params.apply:
            if abs_params.algorithm == "fuller_kapton":
                from dials.algorithms.integration.kapton_correction import (
                    multi_kapton_correction,
                )
            elif abs_params.algorithm == "kapton_2019":
                from dials.algorithms.integration.kapton_2019_correction import (
                    multi_kapton_correction,
                )

            experiments, integrated = multi_kapton_correction(
                experiments, integrated, abs_params.fuller_kapton, logger=logger
            )()

    if params.significance_filter.enable:
        from dials.algorithms.integration.stills_significance_filter import (
            SignificanceFilter,
        )

        sig_filter = SignificanceFilter(params)
        filtered_refls = sig_filter(experiments, integrated)
        accepted_expts = ExperimentList()
        accepted_refls = flex.reflection_table()
        logger.info(
            "Removed %d reflections out of %d when applying significance filter",
            len(integrated) - len(filtered_refls),
            len(integrated),
        )
        for expt_id, expt in enumerate(experiments):
            refls = filtered_refls.select(filtered_refls["id"] == expt_id)
            if len(refls) > 0:
                accepted_expts.append(expt)
                refls["id"] = flex.int(len(refls), len(accepted_expts) - 1)
                accepted_refls.extend(refls)
            else:
                logger.info(
                    "Removed experiment %d which has no reflections left after applying significance filter",
                    expt_id,
                )

        if len(accepted_refls) == 0:
            raise ValueError("No reflections left after applying significance filter")
        experiments = accepted_expts
        integrated = accepted_refls

    from dials.algorithms.indexing.stills_indexer import (
        calc_2D_rmsd_and_displacements,
    )

    rmsd_indexed, _ = calc_2D_rmsd_and_displacements(indexed)
    log_str = f"RMSD indexed (px): {rmsd_indexed:f}\n"
    for i in range(6):
        bright_integrated = integrated.select(
            (
                integrated["intensity.sum.value"]
                / flex.sqrt(integrated["intensity.sum.variance"])
            )
            >= i
        )
        if len(bright_integrated) > 0:
            rmsd_integrated, _ = calc_2D_rmsd_and_displacements(bright_integrated)
        else:
            rmsd_integrated = 0
        log_str += (
            "N reflections integrated at I/sigI >= %d: % 4d, RMSD (px): %f\n"
            % (i, len(bright_integrated), rmsd_integrated)
        )

    for crystal_model in experiments.crystals():
        if hasattr(crystal_model, "get_domain_size_ang"):
            log_str += ". Final ML model: domain size angstroms: {:f}, half mosaicity degrees: {:f}".format(
                crystal_model.get_domain_size_ang(),
                crystal_model.get_half_mosaicity_deg(),
            )

    logger.info(log_str)

    logger.info("")
    logger.info("Time Taken = %f seconds", time.time() - st)
    return experiments, integrated

