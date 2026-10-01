"""Deterministic public example using synthetic inputs only."""
import json

from .adapter import infection_spread_model


def main():
    counts = [10, 20, 5]
    result = infection_spread_model(counts)
    print(json.dumps({
        'counts': counts,
        'contacts': [{'locations': list(pair), 'weight': weight}
                     for pair, weight in result['contacts'].items()],
        'total_contacts': result['total_contacts'],
    }, sort_keys=True))


if __name__ == '__main__':
    main()
