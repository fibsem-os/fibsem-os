from fibsem.microscope import THERMO_API_AVAILABLE
from fibsem.microscopes.tescan import TESCAN_API_AVAILABLE


def main():

    # OpenFIBSEM API
    print("\n\nOpenFIBSEM API:\n")
    try:
        import fibsem
        FIBSEM_AVAILABLE = True
    except ImportError:
        FIBSEM_AVAILABLE = False
    if FIBSEM_AVAILABLE:
        from fibsem.versioning import get_branch, get_version_string

        print(f"OpenFIBSEM v{get_version_string()}")
        branch = get_branch()
        if branch:
            print(f"Branch: {branch}")
        print(f"Installed at: {fibsem.__path__}")
    
    print("-" * 80)
    
    print("Applications:\n")
    try: 
        from fibsem.applications import autolamella
        AUTOLAMELLA_AVAILABLE = True
    except ImportError: 
        AUTOLAMELLA_AVAILABLE = False
    if AUTOLAMELLA_AVAILABLE:
        # getattr rather than direct access: this script exists to report what is
        # installed, so a package that omits __version__ must not abort the report.
        print(f"AutoLamella v{getattr(autolamella, '__version__', 'unknown')}")
        print(f"Installed at: {autolamella.__path__}")
    
    try: 
        import salami
        SALAMI_AVAILABLE = True
    except ImportError:
        SALAMI_AVAILABLE = False
    if SALAMI_AVAILABLE:
        print(f"SALAMI v{getattr(salami, '__version__', 'unknown')}")
        print(f"Installed at: {salami.__path__}")
    print("-" * 80)
    
    # Hardware APIs
    print("Hardware APIs:\n")
    
    # Thermo Fisher API
    print(f"ThermoFisher API {'Available' if THERMO_API_AVAILABLE else 'Not Available'}")
    if THERMO_API_AVAILABLE:
        from fibsem.microscope import version as autoscript_version
        print(f"AutoScript v{autoscript_version}")
    print("-" * 80)

    # Tescan API
    print(f"Tescan API {'Available' if TESCAN_API_AVAILABLE else 'Not Available'}")
    if TESCAN_API_AVAILABLE:
        from fibsem.microscopes.tescan import tescanautomation
        print(f"TescanAutomation v{tescanautomation.__version__}")

if __name__ == "__main__":
    main()