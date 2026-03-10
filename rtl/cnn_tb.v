//============================================================================
// Testbench for CNN Top Level - 1D CNN Accelerator
// Loads 128-sample input from file, feeds sequentially, displays result
//============================================================================

`timescale 1ns/1ps

module cnn_tb;

    //========================================================================
    // Parameters
    //========================================================================
    localparam DATA_WIDTH     = 8;
    localparam OUT_WIDTH      = 16;
    localparam INPUT_LENGTH   = 12;
    localparam INPUT_MEM_DEPTH = 134;
    localparam CLK_PERIOD     = 10;  // 10ns = 100MHz
    
    //========================================================================
    // Testbench signals
    //========================================================================
    reg  clk;
    reg  rst;
    reg  valid_in;
    reg  signed [DATA_WIDTH-1:0] data_in;
    wire valid_out;
    wire [1:0] class_out;
    wire [7:0] confidence;
    wire high_confidence;
    wire early_exit_taken;
    wire [1:0] exit_layer;
    
    // XAI outputs
    wire [3:0] most_important_sample;
    wire [3:0] most_important_filter;
    wire [7:0] importance_score;
    wire [7:0] total_activation;

    // Input data storage
    reg signed [DATA_WIDTH-1:0] input_data [0:INPUT_MEM_DEPTH-1];
    integer i, j;

    // Counters
    reg [7:0] sample_cnt;
    reg [31:0] cycle_cnt;
    reg [1023:0] input_file;

    //========================================================================
    // DUT instantiation
    //========================================================================
    cnn_top #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(32),
        .OUT_WIDTH(OUT_WIDTH),
        .DEBUG_DUMP(1)  // Enable debug dump for parity checking
    ) u_cnn_top (
        .clk(clk),
        .rst(rst),
        .valid_in(valid_in),
        .data_in(data_in),
        .valid_out(valid_out),
        .class_out(class_out),
        .confidence(confidence),
        .high_confidence(high_confidence),
        .early_exit_taken(early_exit_taken),
        .exit_layer(exit_layer),
        // XAI outputs
        .most_important_sample(most_important_sample),
        .most_important_filter(most_important_filter),
        .importance_score(importance_score),
        .total_activation(total_activation)
    );

    //========================================================================
    // Clock generation
    //========================================================================
    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    //========================================================================
    // Load input data from file
    //========================================================================
    initial begin
        input_file = "input_data.mem";
        if ($value$plusargs("INPUT_FILE=%s", input_file)) begin
            $display("Using input file: %0s", input_file);
        end
        $readmemh(input_file, input_data);
    end

    //========================================================================
    // Main test sequence
    //========================================================================
    initial begin
        // Initialize
        rst       = 1'b1;
        valid_in  = 1'b0;
        data_in   = {DATA_WIDTH{1'b0}};
        sample_cnt = 8'd0;
        cycle_cnt = 32'd0;
        
        // Display header
        $display("============================================");
        $display("CNN 1D Accelerator Testbench");
        $display("Hypoglycemia Prediction");
        $display("============================================");
        $display("");
        $display("Time\t| Cycle\t| State");
        $display("--------------------------------------------");
        
        // Reset sequence
        #100;
        rst = 1'b0;
        #20;
        
        // Start feeding input data
        $display("Feeding %d input samples...", INPUT_LENGTH);
        
        for (i = 0; i < INPUT_LENGTH; i = i + 1) begin
            @(posedge clk);
            cycle_cnt = cycle_cnt + 1;
            
            valid_in = 1'b1;
            data_in  = input_data[i];
            sample_cnt = sample_cnt + 8'd1;
            
            $display("%0t\t| %d\t| Feeding sample %d: %d", 
                     $time, cycle_cnt, i, input_data[i]);
        end
        
        // Stop sending data
        @(posedge clk);
        cycle_cnt = cycle_cnt + 1;
        valid_in = 1'b0;
        
        $display("");
        $display("All samples fed. Waiting for result...");
        $display("--------------------------------------------");
        
        // Wait for output (valid_out now includes confidence calculation)
        while (!valid_out) begin
            @(posedge clk);
            cycle_cnt = cycle_cnt + 1;

            // Timeout protection
            if (cycle_cnt > 100000) begin
                $display("ERROR: Timeout waiting for output!");
                $finish;
            end
        end
        
        // Results are ready (class + confidence)
        @(posedge clk);
        
        // Display result
        @(posedge clk);
        $display("");
        $display("============================================");
        $display("RESULT:");
        $display("  Predicted Class: %d", class_out);
        $display("  Confidence: %d/255 (%0d%%)", confidence, (confidence*100)/255);
        $display("  High Confidence: %s", high_confidence ? "YES" : "NO");
        $display("  Early Exit Taken: %s", early_exit_taken ? "YES" : "NO");
        $display("  Exit Layer: %0d", exit_layer);
        $display("  Total Cycles: %d", cycle_cnt);
        $display("--------------------------------------------");
        $display("XAI (Explainable AI):");
        $display("  Most Important Sample: #%d (glucose reading)", most_important_sample);
        $display("  Most Active Filter: #%d", most_important_filter);
        $display("  Importance Score: %d/255", importance_score);
        $display("  Total Activation: %d", total_activation);
        $display("============================================");

        if (class_out == 0) begin
            $display("Prediction: NORMAL (Class 0)");
        end
        else begin
            $display("Prediction: HYPOGLYCEMIA (Class 1)");
        end

        if (high_confidence) begin
            $display(">> CONFIDENCE THRESHOLD MET - Reliable prediction");
        end
        else begin
            $display(">> LOW CONFIDENCE - Consider additional verification");
        end
        
        if (early_exit_taken) begin
            $display(">> EARLY EXIT - Saved cycles with fast inference");
        end
        else begin
            $display(">> FULL NETWORK - All layers executed");
        end
        
        // Run for a few more cycles
        repeat (10) @(posedge clk);
        
        $display("");
        $display("Testbench completed successfully.");
        $finish;
    end

    //========================================================================
    // Monitor output
    //========================================================================
    always @(posedge clk) begin
        if (valid_out) begin
            $display("%0t: OUTPUT VALID - Class = %d", $time, class_out);
        end
    end

    //========================================================================
    // Waveform dump (for debugging)
    //========================================================================
    initial begin
        $dumpfile("cnn_tb.vcd");
        $dumpvars(0, cnn_tb);
    end

endmodule
