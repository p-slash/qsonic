#!/usr/bin/env python
"""Translate a picca delta-extraction .ini file into a qsonic SLURM shell script."""
import argparse
import configparser
import os
import warnings

HEADER = """#!/bin/bash -l
#SBATCH -C cpu
#SBATCH --account={account}
#SBATCH -q {qos}
#SBATCH --nodes=1
#SBATCH --time={time}
#SBATCH --job-name={job_name}

umask 0027
{environment}"""

TURNER24_MEANFLUX = (
    "/dvs_ro/cfs/cdirs/desicollab/science/lya/y1-p1d/"
    "iron-baseline/catalogs/turner24_meanflux.fits"
)
DEFAULTS = dict(
    account="desi", qos="regular", time="2:00:00", job_name="qsonic-Lya",
    environment="source /global/cfs/projectdirs/desi/science/lya/scripts/activate_qsonic.sh",
    fiducial_meanflux=None, out_subdir="Delta",
    nprocs="128", srun="srun -N 1 -n {nprocs} -c 2 qsonic-fit",
    path_from="/global/cfs", path_to="/dvs_ro/cfs",
    rfdwave="0.4", skip="0.0", num_iterations="20", smoothing_scale="0",
    cont_order="1", arms="B R", coadd_arms="before",
    min_rsnr="0.0", min_forestsnr="0.0",
)

# picca keys that are intentionally dropped or have no qsonic counterpart
IGNORED = {
    ("general", "overwrite"),
    ("data", "type"), ("data", "wave solution"),
    ("expected flux", "type"), ("expected flux", "limit var lss"),
    ("expected flux", "fudge value"), ("expected flux", "iter out prefix"),
    ("expected flux", "force stack delta to zero"),
    ("corrections", "num corrections"), ("masks", "num masks"),
}
HANDLED_DATA_TYPES = {"DesiHealpix", "DesiHealpixFast", "DesisimMocks"}


def _f(value):
    return str(float(value))


def _typed_sections(cfg, section, count_key, args_prefix):
    """Return [(type, {arg: value})] for indexed type N / [<prefix> N] entries."""
    if not cfg.has_section(section):
        return []
    n = cfg.getint(section, count_key, fallback=0)
    return [
        (cfg.get(section, f"type {i}"),
         dict(cfg.items(f"{args_prefix} {i}")) if cfg.has_section(f"{args_prefix} {i}") else {})
        for i in range(n)
    ]


def convert_picca_ini_to_qsonic(ini_path, out_path=None, **overrides):
    """Convert a picca ini file to a qsonic shell script.

    Parameters
    ----------
    ini_path : str
        picca configuration file.
    out_path : str, optional
        If given, the script is written there (and made executable).
    **overrides
        Any key of DEFAULTS (e.g. job_name, time, arms, out_subdir,
        path_from=None to disable path rewriting).

    Returns
    -------
    tuple of str
        (header, body, appendix, full script text).
    """
    opt = {**DEFAULTS, **overrides}
    if overrides.get("fiducial_meanflux") is True:
        opt["fiducial_meanflux"] = TURNER24_MEANFLUX

    cfg = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=None)
    cfg.read(ini_path)
    notes = []

    def path(p):
        if opt["path_from"] and p.startswith(opt["path_from"]):
            return opt["path_to"] + p[len(opt["path_from"]):]
        return p

    def get(section, key):
        return cfg.get(section, key, fallback=None)

    data_type = get("data", "type")
    if data_type not in HANDLED_DATA_TYPES:
        notes.append(f"data type '{data_type}' not translated")
    mock = data_type == "DesisimMocks"
    catalog = get("data", "catalogue")

    lines = []
    lines.append("-i " + path(get("data", "input directory").rstrip("/")))
    # output directory is kept as written in the picca ini (no path rewrite)
    out_dir = (get("general", "out dir") or ".").rstrip("/")
    if opt["out_subdir"]:
        out_dir += "/" + opt["out_subdir"].strip("/")
    lines.append("-o " + out_dir)
    lines.append("--skip-resomat" + (" --mock-analysis" if mock else ""))
    if catalog:
        lines.append(f"--catalog {path(catalog)}")
    rfdwave = _f(get("data", "delta lambda")) if get("data", "delta lambda") else opt["rfdwave"]
    lines.append(
        f"--rfdwave {rfdwave} --skip {opt['skip']} --skip-min-pixels "
        f"{get('data', 'minimum number pixels in forest') or 150}")
    lines.append(f"--num-iterations {opt['num_iterations']} --smoothing-scale {opt['smoothing_scale']}")
    lines.append(f"--cont-order {opt['cont_order']}")
    lines.append(f"--wave1 {_f(get('data', 'lambda min'))} --wave2 {_f(get('data', 'lambda max'))}")
    lines.append(f"--forest-w1 {_f(get('data', 'lambda min rest frame'))} "
                 f"--forest-w2 {_f(get('data', 'lambda max rest frame'))}")
    lines.append(f"--arms {opt['arms']}")

    masks = _typed_sections(cfg, "masks", "num masks", "mask arguments")
    corrections = _typed_sections(cfg, "corrections", "num corrections", "correction arguments")
    mask_flags = {"LinesMask": "--sky-mask", "DlaMask": "--dla-mask"}
    order = {"DlaMask": 0, "BalMask": 1, "LinesMask": 2}
    for mtype, margs in sorted(masks, key=lambda m: order.get(m[0], 9)):
        fname = margs.get("filename")
        if mtype in mask_flags:
            lines.append(f"{mask_flags[mtype]} {path(fname)}")
        elif mtype == "BalMask":
            if mock:
                lines.append(f"--bal-mask-catalog {path(fname)}")
            else:
                # qsonic reads BAL info from the main catalogue for real data
                lines = [f"--catalog {path(fname)}" if l.startswith("--catalog") else l for l in lines]
                lines.append("--bal-mask")
        else:
            notes.append(f"mask '{mtype}' not translated")
    lines.append(f'--coadd-arms "{opt["coadd_arms"]}"')

    var = f"--var-fit-eta --eta-varlss {_f(get('expected flux', 'var lss mod'))}" \
        if get("expected flux", "var lss mod") else "--var-fit-eta"
    lines.append(f"{var} --min-rsnr {opt['min_rsnr']} --min-forestsnr {opt['min_forestsnr']}")
    for ctype, cargs in corrections:
        if ctype == "CalibrationCorrection":
            f = path(cargs["filename"])
            lines.append(f"--flux-calibration {f}")
            lines.append(f"--noise-calibration {f}")
        else:
            notes.append(f"correction '{ctype}' not translated")

    ef_type = get("expected flux", "type")
    is_dr16 = bool(ef_type) and ef_type.startswith("Dr16")
    if ef_type == "TrueContinuum":
        if mock:
            lines.append("--true-continuum")
        else:
            notes.append("expected flux type 'TrueContinuum' requires mock data "
                         "(--true-continuum needs --mock-analysis); not translated")
    # picca's Dr16* expected flux stacks to zero unless told otherwise
    if cfg.getboolean("expected flux", "force stack delta to zero", fallback=is_dr16):
        lines.append("--normalize-stacked-flux")

    if opt["fiducial_meanflux"]:
        lines.append(f"--fiducial-meanflux {opt['fiducial_meanflux']}")

    if ef_type not in (None, "TrueContinuum") and not is_dr16:
        notes.append(f"expected flux type '{ef_type}' not translated")
    for section in ("general", "data", "expected flux"):
        for key in cfg.options(section) if cfg.has_section(section) else []:
            if (section, key) not in IGNORED and not _is_used(section, key):
                notes.append(f"[{section}] {key} not translated")

    for n in notes:
        warnings.warn(f"picca->qsonic: {n}")

    nprocs = get("general", "num processors") or opt["nprocs"]
    srun = opt["srun"].format(nprocs=nprocs)
    body = f"{srun} \\\n" + " \\\n".join(lines)
    env = f"\n{opt['environment']}\n" if opt["environment"] else ""
    hdr = HEADER.format(**{**opt, "environment": env})
    apendix = "".join(f"# NOTE: {n}\n" for n in notes) + ("\n" if notes else "")
    text = hdr + "\n" + body + "\n\n" + apendix
    if out_path:
        with open(out_path, "w") as fh:
            fh.write(text)
        os.chmod(out_path, 0o755)
    return hdr, body, apendix, text


_USED = {
    ("data", k) for k in (
        "input directory", "catalogue", "lambda min", "lambda max",
        "lambda min rest frame", "lambda max rest frame",
        "minimum number pixels in forest", "delta lambda")
} | {("expected flux", "var lss mod"), ("general", "out dir"), ("general", "num processors")}


def _is_used(section, key):
    return (section, key) in _USED


def get_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ini")
    p.add_argument("-o", "--output", help="output .sh (default: stdout)")
    p.add_argument("--job-name")
    p.add_argument("--time")
    p.add_argument("--out-subdir", help="subdirectory appended to the picca 'out dir' for -o (default: Delta; pass \"\" for none)")
    p.add_argument("--environment", help="shell line(s) to set up the environment")
    p.add_argument("--fiducial-meanflux", help="option to use Turner24 fiducial mean flux file", action="store_true")
    return p


def main():
    a = get_parser().parse_args()
    ov = {k: v for k, v in (("job_name", a.job_name), ("time", a.time),
                 ("environment", a.environment), ("fiducial_meanflux", a.fiducial_meanflux)) if v}
    if a.out_subdir is not None:
        ov["out_subdir"] = a.out_subdir
    if not a.output:
        warnings.filterwarnings("ignore")

    hdr, body, apendix, text = convert_picca_ini_to_qsonic(a.ini, a.output, **ov)
    if not a.output:
        print(body, end="")


if __name__ == "__main__":
    main()
