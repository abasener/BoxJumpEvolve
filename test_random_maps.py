from maps import load_map

print("Testing Random Map Generation\n")
print("=" * 60)

for i in range(10):
    map_data = load_map('RANDOM', 1400)
    difficulty = map_data['difficulty']
    gaps = len(map_data['gaps'])
    walls = len(map_data['walls'])
    platforms = len(map_data['platforms'])

    print(f"Map {i+1:2d}: Difficulty={difficulty:2d} | Gaps={gaps:2d} | Walls={walls:2d} | Platforms={platforms:2d}")

print("\n" + "=" * 60)
print("Random map generation working!")
