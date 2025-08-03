import argparse
import nmea2xyz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nmea', type=str, required=True, help='path to the input nmea data file.')
    parser.add_argument('--rotp', type=int, required=True, nargs='+', help='index of points to rotate traj to align')
    parser.add_argument('--visz', '-v', action="store_true", help='visualize result trajectory')
    args = parser.parse_args()

    nmea2xyz.convert_nmea_to_xyz(args.nmea, args.rotp, args.visz)

if __name__ == "__main__":
    main()
    