//============================================================================
// Confidence Unit Demo - Shows confidence for different logit pairs
// Tests multiple scenarios: high confidence, low confidence, edge cases
//============================================================================

`timescale 1ns/1ps

module confidence_demo_tb;

    localparam CLK_PERIOD = 10;

    reg  clk, rst, start;
    reg  signed [15:0] logit0, logit1;
    wire [7:0]  confidence;
    wire        high_confidence, done;

    integer test_num;

    // Instantiate confidence unit
    confidence_unit #(
        .LOGIT_WIDTH(16),
        .CONF_WIDTH(8),
        .CONF_THRESHOLD(8'd180)  // 70% threshold
    ) u_conf (
        .clk(clk),
        .rst(rst),
        .start(start),
        .logit0(logit0),
        .logit1(logit1),
        .confidence(confidence),
        .high_confidence(high_confidence),
        .done(done)
    );

    // Clock
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Test task
    task test_confidence;
        input signed [15:0] l0, l1;
        input [7:0] expected_conf;
        begin
            @(posedge clk);
            logit0 = l0;
            logit1 = l1;
            start  = 1;
            
            @(posedge clk);
            start = 0;

            while (!done) @(posedge clk);

            $display("Test %0d: logit0=%4d, logit1=%4d -> conf=%3d/255 (%3d%%), high=%b",
                test_num, l0, l1, confidence, (confidence*100)/255, high_confidence);
            test_num = test_num + 1;
        end
    endtask

    // Main test
    initial begin
        $display("\n");
        $display("╔════════════════════════════════════════════════════════════╗");
        $display("║         CONFIDENCE UNIT DEMO - Various Scenarios          ║");
        $display("╚════════════════════════════════════════════════════════════╝");
        $display("");

        rst = 1; logit0 = 0; logit1 = 0; start = 0; test_num = 1;
        #50; rst = 0; #20;

        $display("--- Scenario 1: HIGH CONFIDENCE (Clear Class 0) ---");
        $display("// Large logit difference: |100 - (-100)| = 200");
        test_confidence(100, -100, 255);
        $display("");

        $display("--- Scenario 2: HIGH CONFIDENCE (Clear Class 1) ---");
        $display("// Large logit difference: |(-100) - 100| = 200");
        test_confidence(-100, 100, 255);
        $display("");

        $display("--- Scenario 3: MEDIUM CONFIDENCE (70% - at threshold) ---");
        $display("// Moderate difference: |85 - 15| = 70, sum = 100");
        test_confidence(85, 15, 179);
        $display("");

        $display("--- Scenario 4: LOW CONFIDENCE (Uncertain) ---");
        $display("// Small difference: |100 - 90| = 10, sum = 190");
        test_confidence(100, 90, 13);
        $display("");

        $display("--- Scenario 5: ZERO CONFIDENCE (Identical logits) ---");
        $display("// No difference: |100 - 100| = 0");
        test_confidence(100, 100, 0);
        $display("");

        $display("--- Scenario 6: BOTH NEGATIVE (High confidence) ---");
        $display("// |(-150) - (-50)| = 100, sum = 200");
        test_confidence(-150, -50, 128);
        $display("");

        $display("--- Scenario 7: ONE LOGIT ZERO (Maximum confidence) ---");
        $display("// |80 - 0| = 80, sum = 80 → 100% confidence");
        test_confidence(80, 0, 255);
        $display("");

        $display("--- Scenario 8: REALISTIC CNN OUTPUT (Class 0 wins) ---");
        $display("// Typical logits from trained model");
        test_confidence(50, -20, 255);
        $display("");

        $display("--- Scenario 9: REALISTIC CNN OUTPUT (Close call) ---");
        $display("// Both positive, small gap");
        test_confidence(45, 35, 32);
        $display("");

        $display("--- Scenario 10: EDGE CASE (Both zero) ---");
        $display("// Neutral confidence for undefined case");
        test_confidence(0, 0, 128);
        $display("");

        $display("╔════════════════════════════════════════════════════════════╗");
        $display("║                        DEMO COMPLETE                       ║");
        $display("╚════════════════════════════════════════════════════════════╝");
        $display("");
        $display("Confidence Threshold: 180/255 = 70%");
        $display("  - high_confidence = 1 when confidence >= 180");
        $display("  - high_confidence = 0 when confidence < 180");
        $display("");

        $finish;
    end

    // VCD dump
    initial begin
        $dumpfile("confidence_demo.vcd");
        $dumpvars(0, confidence_demo_tb);
    end

endmodule
