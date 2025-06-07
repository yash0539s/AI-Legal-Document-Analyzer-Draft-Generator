from backend.services.risk_detector import RiskDetector

def main():
    detector = RiskDetector()

    # Sample texts to test
    texts = [
        "This text has a high risk of fraud.",
        "This is a safe and normal text.",
        "This might be medium risk.",
        ""
    ]

    for text in texts:
        result = detector.detect(text)
        print("Risk Detection for text:", repr(text))
        print(f"Risk: {result['risk']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Severity: {result['severity']}")
        print("-" * 40)

if __name__ == "__main__":
    main()
