import re
import os

def extract_declarations(cu_file):
    """Extract function and class declarations from a .cu file."""
    with open(cu_file, 'r') as file:
        content = file.read()

    # Regex patterns for functions and classes
    function_pattern = re.compile(r'\b(?:__host__|__device__|__global__|void|int|float|double|char|bool|[a-zA-Z_][a-zA-Z0-9_<>]*)\s+\**\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*;?')
    class_pattern = re.compile(r'\bclass\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\{')

    # Find matches
    functions = function_pattern.findall(content)
    classes = class_pattern.findall(content)

    return functions, classes

def generate_header(cu_file, header_file):
    """Generate a header file from a .cu file."""
    functions, classes = extract_declarations(cu_file)

    with open(header_file, 'w') as file:
        guard_macro = os.path.basename(header_file).replace('.', '_').upper()

        file.write(f'#ifndef {guard_macro}\n')
        file.write(f'#define {guard_macro}\n\n')

        file.write('// Function declarations\n')
        for func in functions:
            file.write(func + '\n')
        file.write('\n')

        file.write('// Class declarations\n')
        for cls in classes:
            class_name = re.search(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', cls).group(1)
            file.write(f'class {class_name};\n')
        file.write('\n')

        file.write(f'#endif // {guard_macro}\n')

def main():
    cu_file = input("Enter the .cu file path: ")
    if not cu_file.endswith('.cu'):
        print("Error: Input file must have a .cu extension.")
        return

    header_file = cu_file.replace('.cu', '.h')
    generate_header(cu_file, header_file)
    print(f"Header file generated: {header_file}")

if __name__ == "__main__":
    main()
