# Contribution Guidelines for BioSPPy
Thank you for your interest in contributing to BioSPPy! We appreciate your efforts to help us improve and expand our toolbox. Below are the guidelines for contributing to BioSPPy, whether you are making a code change inside a function, adding a new function to a module, or adding a completely new module.

## How to Contribute
You will need to have a [GitHub account](https://github.com/signup) to contribute to BioSPPy.
1. Fork the Repository

Navigate to the [BioSPPy's](https://github.com/scientisst/BioSPPy) GitHub repository.
Click the "Fork" button in the top right corner to create your own copy of the repository. Now you have a copy of the original repository in your GitHub account.

![](/docs/images/fork_button.png)

2. Clone Your Fork

Clone your forked repository to your local machine:

```bash
git clone https://github.com/yourusername/biosppy.git
cd biosppy
```

Instead of using the terminal, you can also use [GitHub Desktop](https://github.com/apps/desktop) to clone the repository.

![](/docs/images/code_button.png)

3. Create a Branch

Create a new branch for your work. Use a descriptive name for your branch:

```bash
git checkout -b your-feature-name
```

## How to Make Changes

To make changes to BioSPPy, open the project in your preferred code editor or IDE. You should install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

It is recommended to create a virtual environment to manage your dependencies. This way, you can avoid conflicts with other projects. You can use frameworks like `virtualenv` or `conda` to create one.

It's time to make changes! You can add new functions, modify existing functions, or create new modules by editing the source code files.

After making your changes, you should commit them to your local repository:

```bash
git add .
git commit -m "Description of your changes"
```

It is recommended to make small, focused commits with clear descriptions of the changes you have made.

## Opening a Pull Request
To open a PR you need first to push your changes to your remote repository:
```bash
git push origin feature/your-feature-name
```

Then, in your forked repository, you will see a "Contribute" button. Click on it and then select "Open Pull Request". Make sure make a PR to the `main` branch of the `scientisst` repository. 
Please provide a clear title and description for your PR. Include a summary of the changes you have made and any relevant information that will help the maintainers review your code.

![pr_button.png](docs/images/pr_button.png)

## Code Style and Best Practices
Consistency: Follow the existing code style and format. Consistency is key to maintaining readability and maintainability.

Style: Ensure your code adheres to the [PEP 8](https://peps.python.org/pep-0008/) style guide. Use [snake_case](https://en.wikipedia.org/wiki/Snake_case) for function and variable names.

Docstrings: Use docstrings to document your functions and modules. Include descriptions for parameters and return values. Follow the [numpydoc format](https://numpydoc.readthedocs.io/en/latest/format.html).

Comments: Comment your code where necessary to explain complex logic.

Dependencies: Avoid adding new dependencies unless absolutely necessary.

![code_formatting.png](docs/images/code_formatting.png)

## Getting Help
If you have any questions or need assistance, feel free to open an issue on the GitHub repository or reach out to the project maintainers.

Thank you for contributing to BioSPPy!