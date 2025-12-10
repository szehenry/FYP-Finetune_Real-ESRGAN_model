"""
Update image paths from OneDrive to D drive
"""

from pathlib import Path

def update_image_list(list_file, old_path, new_path):
    """
    Update paths in image list file
    
    Args:
        list_file: Path to list file
        old_path: Old path to replace
        new_path: New path
    """
    print(f"\nProcessing: {list_file}")
    
    # Read file
    with open(list_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Update paths
    updated_lines = []
    updated_count = 0
    
    for line in lines:
        line = line.strip()
        if line and old_path in line:
            new_line = line.replace(old_path, new_path)
            updated_lines.append(new_line + '\n')
            updated_count += 1
        elif line:
            updated_lines.append(line + '\n')
    
    # Write back
    with open(list_file, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    
    print(f"  Updated {updated_count} paths")
    return updated_count

def main():
    # Set paths
    old_path = r"C:\Users\henry\OneDrive - The Hong Kong Polytechnic University\Y4_SEM1\FYP_Images"
    new_path = r"D:\FYP_Images"
    
    split_dir = Path("own_Windows_data_split_results")
    
    print("=" * 80)
    print("Updating Image Paths")
    print("=" * 80)
    print(f"Old path: {old_path}")
    print(f"New path: {new_path}")
    
    total_updated = 0
    
    # Update all list files
    for split_name in ['train', 'val', 'test']:
        list_file = split_dir / f'{split_name}_list.txt'
        if list_file.exists():
            count = update_image_list(list_file, old_path, new_path)
            total_updated += count
        else:
            print(f"\nWarning: {list_file} does not exist")
    
    print("\n" + "=" * 80)
    print(f"Complete! Updated {total_updated} paths in total")
    print("=" * 80)

if __name__ == '__main__':
    main()

