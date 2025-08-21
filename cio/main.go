package main

import (
	"bufio"
	"fmt"
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

	fmt.Print("Enter the amount of money to allocate (ZAR): ")
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)

	amount, err := strconv.ParseFloat(input, 64)
	if err != nil {
		fmt.Println("Invalid input. Please enter a valid number.")
		return
	}

	fmt.Println("\nSuper 8 Portfolio Capital Allocations (CIO Strategy):")
	fmt.Println("--------------------------------------------------")
	fmt.Printf("%-5s %-12s\n", "Stock", "Amount (ZAR)")
	fmt.Println("--------------------------------------------------")
	total := 0.0
	for _, stock := range stocks {
		allocation := (stock.Weight / 100) * amount
		total += allocation
		fmt.Printf("%-5s %-12.2f\n", stock.Ticker, allocation)
	}
	fmt.Println("--------------------------------------------------")
	fmt.Printf("Total Allocated: %.2f ZAR\n", total)
}
