import os

def create_project_structure():
    """
    Create the project directory structure.
    """
    # Define the directory structure
    directories = [
        'data/raw',
        'data/processed',
        'data/external',
        'models',
        'notebooks',
        'reports/figures',
        'src/data',
        'src/features',
        'src/models',
        'src/visualization',
        'tests'
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    # Create __init__.py files in src directories
    src_dirs = [
        'src',
        'src/data',
        'src/features',
        'src/models',
        'src/visualization'
    ]
    
    for directory in src_dirs:
        init_file = os.path.join(directory, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Initialize package\n\n')
    
    print("Project structure created successfully!")

if __name__ == "__main__":
    create_project_structure() 