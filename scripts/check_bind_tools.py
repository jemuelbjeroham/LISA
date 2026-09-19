import inspect

from langchain_nvidia_ai_endpoints import ChatNVIDIA


def main() -> None:
    signature = inspect.signature(ChatNVIDIA.bind_tools)

    print("ChatNVIDIA.bind_tools signature:")
    print(signature)

    print("\nDetailed parameters:")
    for name, parameter in signature.parameters.items():
        print(f"{name}:")
        print(f"  kind:    {parameter.kind}")
        print(f"  default: {parameter.default}")
        print(f"  annotation: {parameter.annotation}")
        print()


if __name__ == "__main__":
    main()