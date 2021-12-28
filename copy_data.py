import shutil
import os

# TODO: no idea Rompers_Jumpsuits is upper or lower

UPPER_LIST = [
	"Blouses_Shirts", "Cardigans", "Graphic_Tees", "Jackets_Coats", "Sweaters", "Sweatshirts_Hoodies", "Tees_Tanks", "Rompers_Jumpsuits",
	"Jackets_Vests", "Shirts_Polos", "Suiting"
]
LOWER_LIST = [
	"Denim", "Dresses", "Leggings", "Pants", "Shorts", "Skirts"
]


def copy_files():
	with open(f"dataset/Eval/list_eval_partition.txt", "r") as fi:
		lines = fi.readlines()
		content_lines = lines[2:]
		for line in content_lines:
			path, item_id, status = line.strip().split()
			path_parts = path.split("/")
			clothes_position = "upper" if path_parts[2] in UPPER_LIST else "lower"
			dest_path = fr'{os.getcwd()}/dataset/{status}/{clothes_position}/{item_id}_{path_parts[-1]}'
			print('->', dest_part)
			shutil.copyfile(fr"dataset/{path}", dest_path)
			# dest_part = fr'{os.getcwd()}/dataset_{clothes_position}/{status}/{path_parts[2]}/{item_id}_{path_parts[-1]}'
			# print('-->', dest_part)
			# shutil.copyfile(fr"dataset/{path}", dest_part)


if __name__ == '__main__':
	copy_files()
