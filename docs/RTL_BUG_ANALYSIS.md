The RTL accumulation bug is likely due to Verilog non-blocking assignment semantics.

Current code:
```verilog
acc <= acc + (dezp * weight);
```

In Verilog, non-blocking assignments (`<=`) schedule the update for the END of the time step.
So if we do:
```
acc <= acc + term1;  // schedules acc_new = acc_old + term1
acc <= acc + term2;  // schedules acc_new = acc_old + term2 (OVERWRITES!)
```

The second assignment OVERWRITES the first, not accumulates!

To fix, we need to either:
1. Use a temporary variable and assign once
2. Or restructure the accumulation

The correct pattern for accumulation in a loop is:
```verilog
always @(posedge clk) begin
    if (reset) begin
        acc <= 0;
    end else if (accumulate) begin
        acc <= acc + term;  // This is fine IF it's only done once per cycle
    end
end
```

But in the RTL, we're doing multiple `acc <= acc + ...` in the same always block
for different values of conv_k. This doesn't work because they all use the SAME acc_old!

Solution: Accumulate in a combinatorial temporary, then assign to acc once.

OR: Only do one multiply-accumulate per cycle and use conv_k to track progress.

The current RTL tries to do all 3 MACs in sequence within the same state, but the
non-blocking assignments don't work that way.
