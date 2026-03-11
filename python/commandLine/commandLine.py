class ArgumentException(Exception):
    def __init__(self, value, message = "Command line argument is missing!"):
        self.value = value
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} {self.value}"


def get_help(program_name):
    help_text = f"""
Usage: {program_name} [option]
Required arguments for algorithm execution:
--in, --input <input_dir>     Sets the directory where the input files will be searched for to <input_dir>.
--out, --output <output_dir>  Sets the directory where the results will be saved to <output_dir>.

Extra arguments available:
-h, --help                    Display this information and exit."""
    
    print(help_text)

def get_target_dirs(argv):
    input_dir = ""
    output_dir = ""

    for i in range(len(argv)):

        try:
            if (argv[i] == "--in" or argv[i] == "--input"):
                input_dir = argv[i+1]

            elif (argv[i] == "--out" or argv[i] == "--output"):
                output_dir = argv[i+1]
            
            elif (argv[i] == "-h" or argv[i] == "--help"):
                get_help(argv[0].split("/")[-1])
                exit()

        except IndexError:
            raise ArgumentException(argv[i] + " <directory_path>")
    
    if len(input_dir) == 0:
        raise ArgumentException("--in <input_directory_path>")
    if len(output_dir) == 0:
        raise ArgumentException("--out <output_directory_path>")
    
    return input_dir, output_dir