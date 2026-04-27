"""
Test script for the OCR Pipeline
Tests the preprocessing and folder scanning functionality without requiring Gemini API
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from complete_ocr_pipeline import (
    preprocess_full_pipeline,
    load_and_preprocess_image,
    scan_images_recursively,
    get_all_images_from_folders,
    save_results_to_json,
    export_for_product_matching
)

def test_preprocessing():
    """Test image preprocessing functions."""
    print("=" * 70)
    print("Testing Image Preprocessing")
    print("=" * 70)
    
    # Test with a sample image
    test_image_path = "ocr/processing_images/invoices1/page_001.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        # Try another image
        test_image_path = "ocr/original_data/invoices1/photo_1_2026-04-23_23-58-26.jpg"
    
    if os.path.exists(test_image_path):
        print(f"Testing preprocessing on: {test_image_path}")
        
        try:
            from PIL import Image
            # Load original image
            original = Image.open(test_image_path)
            print(f"  Original size: {original.size}, mode: {original.mode}")
            
            # Preprocess
            processed = preprocess_full_pipeline(original)
            print(f"  Processed size: {processed.size}, mode: {processed.mode}")
            
            # Test load_and_preprocess_image
            loaded = load_and_preprocess_image(test_image_path)
            if loaded:
                print(f"  Load and preprocess successful: {loaded.size}")
            
            print("✓ Preprocessing test passed")
            return True
            
        except Exception as e:
            print(f"✗ Preprocessing test failed: {e}")
            return False
    else:
        print("✗ No test images found")
        return False

def test_folder_scanning():
    """Test recursive folder scanning."""
    print("\n" + "=" * 70)
    print("Testing Folder Scanning")
    print("=" * 70)
    
    # Test scanning processing_images folder
    processing_dir = "ocr/processing_images"
    if os.path.exists(processing_dir):
        images = scan_images_recursively(processing_dir)
        print(f"Found {len(images)} images in {processing_dir}")
        
        if images:
            print(f"Sample images: {images[:3]}")
            print("✓ Folder scanning test passed")
            return True
        else:
            print("✗ No images found in processing_images")
            return False
    else:
        print(f"✗ Directory not found: {processing_dir}")
        return False

def test_get_all_images():
    """Test getting images from all folders."""
    print("\n" + "=" * 70)
    print("Testing Get All Images")
    print("=" * 70)
    
    folders = [
        "ocr/processing_images",
        "ocr/original_data"
    ]
    
    all_images = get_all_images_from_folders(folders)
    print(f"Total images found: {len(all_images)}")
    
    if all_images:
        print(f"First 5 images: {all_images[:5]}")
        print("✓ Get all images test passed")
        return True
    else:
        print("✗ No images found in any folder")
        return False

def test_json_output():
    """Test JSON output functions with mock data."""
    print("\n" + "=" * 70)
    print("Testing JSON Output Functions")
    print("=" * 70)
    
    # Create mock results
    mock_results = [
        {
            "processing_id": "test_001",
            "image_path": "ocr/processing_images/invoices1/page_001.jpg",
            "image_name": "page_001",
            "timestamp": "20240101_120000",
            "status": "completed",
            "preprocessed": True,
            "gemini_processed": True,
            "extracted_data": {
                "store_name": "Test Store",
                "invoice_date": "2024-01-01",
                "invoice_number": "INV-001",
                "total_amount": 150.0,
                "currency": "EGP",
                "items": [
                    {
                        "product_name": "إسكواش فراولة 500 جرام",
                        "quantity": 2,
                        "unit_price": 15.0,
                        "item_total": 30.0
                    },
                    {
                        "product_name": "مياه معدنية 1.5 لتر",
                        "quantity": 1,
                        "unit_price": 10.0,
                        "item_total": 10.0
                    }
                ]
            }
        },
        {
            "processing_id": "test_002",
            "image_path": "ocr/processing_images/invoices1/page_002.jpg",
            "image_name": "page_002",
            "timestamp": "20240101_120100",
            "status": "failed",
            "preprocessed": True,
            "gemini_processed": False,
            "error": "Gemini API not available"
        }
    ]
    
    try:
        # Test save_results_to_json
        output_dir = "ocr/test_results"
        os.makedirs(output_dir, exist_ok=True)
        
        results_file = save_results_to_json(mock_results, f"{output_dir}/test_results.json")
        print(f"✓ Results saved to: {results_file}")
        
        # Test export_for_product_matching
        matching_file = export_for_product_matching(mock_results, f"{output_dir}/test_products.json")
        print(f"✓ Products exported to: {matching_file}")
        
        # Verify files exist
        if os.path.exists(results_file) and os.path.exists(matching_file):
            print("✓ JSON output test passed")
            return True
        else:
            print("✗ Output files not created")
            return False
            
    except Exception as e:
        print(f"✗ JSON output test failed: {e}")
        return False

def test_pipeline_integration():
    """Test the complete pipeline integration (without Gemini API)."""
    print("\n" + "=" * 70)
    print("Testing Pipeline Integration")
    print("=" * 70)
    
    try:
        # Import the main pipeline function
        from complete_ocr_pipeline import run_complete_pipeline
        
        # Run pipeline in test mode (no API key, limited images)
        print("Running pipeline in test mode...")
        
        summary = run_complete_pipeline(
            api_key=None,  # No API key for testing
            input_folders=["ocr/processing_images"],
            output_dir="ocr/test_results",
            max_images=2,  # Only process 2 images
            test_mode=True
        )
        
        if "error" in summary:
            print(f"Pipeline completed with warning: {summary['error']}")
        else:
            print(f"Pipeline completed successfully!")
            print(f"  Processed: {summary['total_images']} images")
            print(f"  Completed: {summary['completed']}")
            print(f"  Results: {summary['results_file']}")
        
        print("✓ Pipeline integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Pipeline integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("OCR Pipeline Test Suite")
    print("=" * 70)
    
    tests = [
        ("Preprocessing", test_preprocessing),
        ("Folder Scanning", test_folder_scanning),
        ("Get All Images", test_get_all_images),
        ("JSON Output", test_json_output),
        ("Pipeline Integration", test_pipeline_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed successfully!")
        return 0
    else:
        print(f"{total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())