package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

func main() {
	stocks := []struct {
		Ticker string
		Weight float64
	}{
		{"GLN", 5.70}, {"PRX", 12.66}, {"AGL", 7.34}, {"SOL", 6.06},
		{"NED", 4.64}, {"MTN", 5.92}, {"PPE", 6.37}, {"CTA", 10.08},
		{"EXX", 7.19}, {"PMR", 2.79}, {"RNI", 5.48}, {"INP", 2.00},
		{"ABG", 4.55}, {"RDF", 2.52}, {"APN", 6.10}, {"BVT", 4.41},
		{"REM", 6.04},
	}

	reader := bufio.NewReader(os.Stdin)

	// Input validation loop
	var amount float64
	for {
		fmt.Print("Enter the amount of money to allocate (ZAR): ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		// Handle empty input
		if input == "" {
			fmt.Println("Error: Input cannot be empty. Please enter a valid number.")
			continue
		}

		// Parse and validate amount
		var err error
		amount, err = strconv.ParseFloat(input, 64)
		if err != nil {
			fmt.Println("Error: Invalid number format. Please enter a valid decimal number (e.g., 1000 or 1000.50).")
			continue
		}
		if amount < 0 {
			fmt.Println("Error: Amount cannot be negative. Please enter a positive number.")
			continue
		}
		if amount == 0 {
			fmt.Println("Error: Amount must be greater than zero.")
			continue
		}
		break
	}

	// Convert to cents for precise calculation
	amountCents := int64(math.Round(amount * 100))

	// Calculate total weight (should be ~100% but may have rounding errors)
	totalWeight := 0.0
	for _, s := range stocks {
		totalWeight += s.Weight
	}

	// Allocate in cents to avoid floating-point errors
	allocations := make([]int64, len(stocks))
	totalAllocatedCents := int64(0)

	for i, stock := range stocks {
		// Calculate ideal allocation in cents
		idealCents := (stock.Weight / totalWeight) * float64(amountCents)
		allocations[i] = int64(math.Round(idealCents))
		totalAllocatedCents += allocations[i]
	}

	// Adjust last stock to fix rounding discrepancies
	if diff := amountCents - totalAllocatedCents; diff != 0 {
		allocations[len(allocations)-1] += diff
	}

	// Display results
	fmt.Println("\nSuper 8 Portfolio Capital Allocations (CIO Strategy):")
	fmt.Println("--------------------------------------------------")
	fmt.Printf("%-5s %-12s\n", "Stock", "Amount (ZAR)")
	fmt.Println("--------------------------------------------------")
	for i, stock := range stocks {
		amountZAR := float64(allocations[i]) / 100.0
		fmt.Printf("%-5s %-12.2f\n", stock.Ticker, amountZAR)
	}
	fmt.Println("--------------------------------------------------")
	fmt.Printf("Total Allocated: %.2f ZAR\n", float64(amountCents)/100.0)
	fmt.Printf("(Note: Original input rounded to nearest cent: %.2f ZAR)\n", amount)
}
