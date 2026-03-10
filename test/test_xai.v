//============================================================================
// XAI Module Unit Testbench
// Tests activation_buffer and xai_scanner modules
//============================================================================

`timescale 1ns/1ps

module test_xai;

    //========================================================================
    // Parameters
    //========================================================================
    localparam DATA_WIDTH   = 8;
    localparam NUM_FILTERS  = 8;
    localparam SEQ_LEN      = 12;
    localparam CLK_PERIOD   = 10;

    //========================================================================
    // Testbench signals
    //========================================================================
    reg  clk;
    reg  rst;
    
    // Activation buffer write
    reg  write_en;
    reg [3:0] write_filter;
    reg [3:0] write_seq;
    reg  signed [DATA_WIDTH-1:0] write_data;
    
    // Activation buffer read / XAI scanner read
    wire read_en;
    wire [3:0] read_filter;
    wire [3:0] read_seq;
    reg  signed [DATA_WIDTH-1:0] read_data_tb;  // Testbench version for manual reads
    wire signed [DATA_WIDTH-1:0] read_data;
    
    // XAI scanner control
    reg  xai_start;
    wire xai_done;
    wire [3:0] most_important_sample;
    wire [3:0] most_important_filter;
    wire [7:0] importance_score;
    wire [7:0] total_activation;
    wire [7:0] avg_activation;
    wire buffer_ready;

    //========================================================================
    // DUT Instances
    //========================================================================
    
    // For testing, we'll use two separate read interfaces:
    // - tb_read_* for testbench manual verification
    // - xai_read_* for XAI scanner
    
    reg tb_read_en;
    reg [3:0] tb_read_filter;
    reg [3:0] tb_read_seq;
    wire signed [DATA_WIDTH-1:0] tb_read_data;
    
    activation_buffer #(
        .NUM_FILTERS(NUM_FILTERS),
        .SEQ_LEN(SEQ_LEN),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_activation_buffer (
        .clk(clk),
        .rst(rst),
        .write_en(write_en),
        .write_filter(write_filter),
        .write_seq(write_seq),
        .write_data(write_data),
        .read_en(tb_read_en | read_en),  // Combined read enable
        .read_filter(tb_read_en ? tb_read_filter : read_filter),
        .read_seq(tb_read_en ? tb_read_seq : read_seq),
        .read_data(read_data),
        .buffer_ready(buffer_ready)
    );
    
    xai_scanner #(
        .NUM_FILTERS(NUM_FILTERS),
        .SEQ_LEN(SEQ_LEN),
        .DATA_WIDTH(DATA_WIDTH),
        .SCORE_WIDTH(8)
    ) u_xai_scanner (
        .clk(clk),
        .rst(rst),
        .start(xai_start),
        .done(xai_done),
        .read_en(read_en),
        .read_filter(read_filter),
        .read_seq(read_seq),
        .read_data(read_data),
        .most_important_sample(most_important_sample),
        .most_important_filter(most_important_filter),
        .importance_score(importance_score),
        .total_activation(total_activation),
        .avg_activation(avg_activation)
    );

    //========================================================================
    // Clock generation
    //========================================================================
    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    //========================================================================
    // Test data - predefined activation map
    // We'll create a pattern where sample #7, filter #5 has the max value
    //========================================================================
    reg signed [DATA_WIDTH-1:0] test_activations [0:95];  // 12 × 8
    
    integer i, f, s;
    integer error_count;

    //========================================================================
    // Test sequence
    //========================================================================
    initial begin
        $display("============================================");
        $display("XAI Module Unit Test");
        $display("============================================");
        $display("");
        
        error_count = 0;
        
        // Initialize
        rst         = 1'b1;
        write_en    = 1'b0;
        write_data  = 8'd0;
        write_filter = 4'd0;
        write_seq   = 4'd0;
        tb_read_en  = 1'b0;
        tb_read_filter = 4'd0;
        tb_read_seq = 4'd0;
        xai_start   = 1'b0;
        
        #100;
        rst = 1'b0;
        #20;
        
        $display("Test 1: Write test pattern to activation buffer");
        $display("--------------------------------------------");
        
        // Create test pattern: max at sample #7, filter #5
        for (s = 0; s < SEQ_LEN; s = s + 1) begin
            for (f = 0; f < NUM_FILTERS; f = f + 1) begin
                if (s == 7 && f == 5)
                    test_activations[s * NUM_FILTERS + f] = 8'd127;  // Max INT8 value
                else if (s == 3 && f == 2)
                    test_activations[s * NUM_FILTERS + f] = 8'd80;   // Second highest
                else
                    test_activations[s * NUM_FILTERS + f] = $random % 50;  // Random 0-49
            end
        end
        
        // Write all 96 activations to buffer
        for (i = 0; i < 96; i = i + 1) begin
            @(posedge clk);
            write_en    <= 1'b1;
            write_seq   <= i / NUM_FILTERS;        // seq = i / 8
            write_filter<= i % NUM_FILTERS;        // filter = i % 8
            write_data  <= test_activations[i];
        end
        
        @(posedge clk);
        write_en <= 1'b0;
        
        $display("Wrote 96 activation values to buffer");
        $display("");
        
        // Test random reads
        $display("Test 2: Verify random reads from buffer");
        $display("--------------------------------------------");
        
        // Read from sample #7, filter #5 (should be 127)
        @(posedge clk);
        tb_read_en   <= 1'b1;
        tb_read_seq  <= 4'd7;
        tb_read_filter <= 4'd5;
        
        @(posedge clk);
        @(posedge clk);  // Extra cycle for read data to appear
        if ($signed(read_data) === 8'd127) begin
            $display("PASS: Read activation[7][5] = %0d (expected 127)", read_data);
        end else begin
            $display("FAIL: Read activation[7][5] = %0d (expected 127)", read_data);
            error_count = error_count + 1;
        end
        
        // Read from sample #3, filter #2 (should be 80)
        @(posedge clk);
        tb_read_seq  <= 4'd3;
        tb_read_filter <= 4'd2;
        
        @(posedge clk);
        @(posedge clk);  // Extra cycle for read data to appear
        if ($signed(read_data) === 8'd80) begin
            $display("PASS: Read activation[3][2] = %0d (expected 80)", read_data);
        end else begin
            $display("FAIL: Read activation[3][2] = %0d (expected 80)", read_data);
            error_count = error_count + 1;
        end
        
        tb_read_en <= 1'b0;
        $display("");
        
        // Test XAI scanner
        $display("Test 3: Run XAI scanner to find max activation");
        $display("--------------------------------------------");
        
        @(posedge clk);
        xai_start <= 1'b1;
        
        @(posedge clk);
        xai_start <= 1'b0;
        
        $display("XAI scanner running... (96 cycles + overhead)");
        
        // Wait for scanner to complete
        while (!xai_done) begin
            @(posedge clk);
        end
        
        @(posedge clk);
        $display("");
        $display("XAI Results:");
        $display("  Most Important Sample: #%0d", most_important_sample);
        $display("  Most Active Filter: #%0d", most_important_filter);
        $display("  Importance Score: %0d", importance_score);
        $display("  Total Activation: %0d", total_activation);
        $display("");
        
        // Verify results
        if (most_important_sample === 4'd7) begin
            $display("PASS: Most important sample = #7");
        end else begin
            $display("FAIL: Most important sample = #%0d (expected #7)", most_important_sample);
            error_count = error_count + 1;
        end
        
        if (most_important_filter === 4'd5) begin
            $display("PASS: Most active filter = #5");
        end else begin
            $display("FAIL: Most active filter = #%0d (expected #5)", most_important_filter);
            error_count = error_count + 1;
        end
        
        if ($signed(importance_score) >= 8'd127) begin
            $display("PASS: Importance score = %0d (expected >= 127)", importance_score);
        end else begin
            $display("FAIL: Importance score = %0d (expected >= 127)", importance_score);
            error_count = error_count + 1;
        end
        
        $display("");
        
        // Test 4: Edge case - all zeros
        $display("Test 4: Edge case - all zero activations");
        $display("--------------------------------------------");
        
        // Write zeros
        for (i = 0; i < 96; i = i + 1) begin
            @(posedge clk);
            write_en    <= 1'b1;
            write_seq   <= i / NUM_FILTERS;
            write_filter<= i % NUM_FILTERS;
            write_data  <= 8'd0;
        end
        
        @(posedge clk);
        write_en <= 1'b0;
        
        @(posedge clk);
        xai_start <= 1'b1;
        
        @(posedge clk);
        xai_start <= 1'b0;
        
        while (!xai_done) begin
            @(posedge clk);
        end
        
        @(posedge clk);
        $display("XAI Results (all zeros):");
        $display("  Most Important Sample: #%0d", most_important_sample);
        $display("  Most Active Filter: #%0d", most_important_filter);
        $display("  Importance Score: %0d", importance_score);
        
        if (importance_score === 8'd0 || importance_score === 8'd128) begin
            $display("PASS: Zero/neutral importance for all-zero input");
        end else begin
            $display("INFO: Importance score = %0d (implementation dependent)", importance_score);
        end
        
        $display("");
        
        // Summary
        $display("============================================");
        $display("TEST SUMMARY");
        $display("============================================");
        if (error_count === 0) begin
            $display("All tests PASSED!");
        end else begin
            $display("%0d test(s) FAILED", error_count);
        end
        $display("============================================");
        
        $finish;
    end

    //========================================================================
    // Waveform dump
    //========================================================================
    initial begin
        $dumpfile("test_xai.vcd");
        $dumpvars(0, test_xai);
    end

endmodule
