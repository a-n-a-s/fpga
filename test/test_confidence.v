//============================================================================
// Testbench for Confidence Unit
// Verifies confidence calculation for various logit pairs
//============================================================================

`timescale 1ns/1ps

module confidence_unit_tb;

    //========================================================================
    // Parameters
    //========================================================================
    localparam LOGIT_WIDTH = 16;
    localparam CONF_WIDTH  = 8;
    localparam CLK_PERIOD  = 10;

    //========================================================================
    // Testbench signals
    //========================================================================
    reg  clk;
    reg  rst;
    reg  start;
    reg  signed [LOGIT_WIDTH-1:0] logit0;
    reg  signed [LOGIT_WIDTH-1:0] logit1;
    wire [CONF_WIDTH-1:0]  confidence;
    wire high_confidence;
    wire done;

    integer test_count;
    integer pass_count;

    //========================================================================
    // DUT instantiation
    //========================================================================
    confidence_unit #(
        .LOGIT_WIDTH(LOGIT_WIDTH),
        .CONF_WIDTH(CONF_WIDTH),
        .CONF_THRESHOLD(8'd180)  // 70% threshold
    ) u_confidence_unit (
        .clk(clk),
        .rst(rst),
        .start(start),
        .logit0(logit0),
        .logit1(logit1),
        .confidence(confidence),
        .high_confidence(high_confidence),
        .done(done)
    );

    //========================================================================
    // Clock generation
    //========================================================================
    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    //========================================================================
    // Test tasks
    //========================================================================
    task run_test;
        input signed [15:0] l0;
        input signed [15:0] l1;
        input [7:0]         expected_conf;
        input               expected_high;
        reg [7:0]           actual_conf;
        reg                 actual_high;
        begin
            @(posedge clk);
            logit0 = l0;
            logit1 = l1;
            start  = 1'b1;
            
            @(posedge clk);
            start = 1'b0;

            // Wait for done
            while (!done) begin
                @(posedge clk);
            end

            actual_conf = confidence;
            actual_high = high_confidence;

            // Check results
            if (actual_conf == expected_conf && actual_high == expected_high) begin
                $display("PASS: logit0=%0d, logit1=%0d -> conf=%0d (%0d%%), high=%b",
                    l0, l1, actual_conf, (actual_conf*100)/255, actual_high);
                pass_count = pass_count + 1;
            end
            else begin
                $display("FAIL: logit0=%0d, logit1=%0d -> conf=%0d (expected %0d), high=%b (expected %b)",
                    l0, l1, actual_conf, expected_conf, actual_high, expected_high);
            end
            test_count = test_count + 1;
        end
    endtask

    //========================================================================
    // Main test sequence
    //========================================================================
    initial begin
        $display("============================================");
        $display("Confidence Unit Testbench");
        $display("============================================");
        $display("");

        // Initialize
        rst       = 1'b1;
        start     = 1'b0;
        logit0    = 16'd0;
        logit1    = 16'd0;
        test_count = 0;
        pass_count = 0;

        // Reset
        #50;
        rst = 1'b0;
        #20;

        $display("Running tests...\n");

        // Test 1: High confidence, class 0 wins
        // |logit0 - logit1| = |100 - (-100)| = 200
        // |logit0| + |logit1| = 100 + 100 = 200
        // confidence = (200/200) * 255 = 255
        run_test(100, -100, 8'd255, 1'b1);

        // Test 2: High confidence, class 1 wins
        run_test(-100, 100, 8'd255, 1'b1);

        // Test 3: Medium confidence (70%)
        // |50 - (-20)| = 70, |50| + |20| = 70
        // confidence = (70/70) * 255 = 255... let me recalculate
        // Actually: |50 - (-20)| = 70, |50| + |-20| = 70
        // confidence = 255
        run_test(50, -20, 8'd255, 1'b1);

        // Test 4: Low confidence (similar logits)
        // |100 - 90| = 10, |100| + |90| = 190
        // confidence = (10/190) * 255 ≈ 13
        run_test(100, 90, 8'd13, 1'b0);

        // Test 5: Zero confidence (identical logits)
        run_test(100, 100, 8'd0, 1'b0);

        // Test 6: Medium confidence ~50%
        // |80 - 0| = 80, |80| + |0| = 80
        // confidence = 255 (one logit is zero)
        run_test(80, 0, 8'd255, 1'b1);

        // Test 7: Negative logits, high confidence
        run_test(-150, -50, 8'd64, 1'b0);  // |−150−(−50)|=100, |−150|+|−50|=200, 100/200*255=128

        // Test 8: Threshold boundary (around 70% = 178/255)
        // Need: diff/sum ≈ 0.7, e.g., diff=70, sum=100
        // logit0=85, logit1=15: diff=70, sum=100
        run_test(85, 15, 8'd179, 1'b1);  // Just above threshold

        // Test 9: Just below threshold
        // logit0=80, logit1=20: diff=60, sum=100, conf=153
        run_test(80, 20, 8'd153, 1'b0);  // Below threshold (180)

        // Test 10: Both zero (edge case)
        run_test(0, 0, 8'd128, 1'b0);  // Neutral confidence

        $display("");
        $display("============================================");
        $display("Test Summary: %0d/%0d tests passed", pass_count, test_count);
        $display("============================================");

        if (pass_count == test_count) begin
            $display("SUCCESS: All tests passed!");
        end
        else begin
            $display("FAILURE: %0d tests failed", test_count - pass_count);
        end

        $finish;
    end

    //========================================================================
    // Waveform dump
    //========================================================================
    initial begin
        $dumpfile("confidence_unit_tb.vcd");
        $dumpvars(0, confidence_unit_tb);
    end

endmodule
