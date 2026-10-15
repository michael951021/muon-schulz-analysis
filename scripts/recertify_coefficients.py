"""Update certificates analytically without changing fitted coefficients."""
import json
from nsshape.coefficients import TABLE_PATH
from nsshape.polynomial import reachable_bound, invariance_violation, positivity_violation, band_extent


def main():
    payload = json.loads(TABLE_PATH.read_text())
    payload['certificate_revision'] = 'analytic-extrema-v2'
    payload['positivity_margin_definition'] = 'min P(x)/x on [0,B]'
    for entry in payload['entries']:
        if entry['coeffs'] is None:
            for key in ('critical_point', 'invariance_margin', 'positivity_margin'):
                entry[key] = None
            continue
        coeffs = tuple(entry['coeffs'])
        bound = reachable_bound(coeffs)
        entry['invariance_bound'] = bound
        entry['invariance_margin'] = -invariance_violation(coeffs, bound)
        entry['positivity_margin'] = -positivity_violation(coeffs, bound)
        extent = band_extent(coeffs, entry['gamma'], entry['n_steps'])
        entry['sigma_min'] = extent.sigma_min
        entry['scale'] = extent.scale
        entry['achieved_half_width'] = extent.achieved_half_width
        assert entry['invariance_margin'] >= -1e-10
        assert entry['positivity_margin'] > 0
    TABLE_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
