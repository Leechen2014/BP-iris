package backp ;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Scanner;
public class BPNN {
	/**
	 * @param args
	 */
	
	int inputnodenum=4;
	int hidennodenum=3;
	int outputnodenum=3;
	int maxstep=5000;
	int step;
	double theta=0.5;
	double lambda=1.0;
	double alpha=0.01;
	double eta=0.5;
	double v[][];
	double w[][];
	double inputdata[];
	double hidendata[];
	double outputdata[];
	double rightoutput[];
	double deltav[][][];
	double deltaw[][][];
	HashMap<String, Integer> map;
	String [] classname;
	
	void init()
	{
		v=new double[inputnodenum][hidennodenum];
		w=new double[hidennodenum][outputnodenum];
		deltav=new double[maxstep+1][inputnodenum][hidennodenum];//0
		deltaw=new double[maxstep+1][hidennodenum][outputnodenum];
		inputdata=new double[inputnodenum];
		hidendata=new double[hidennodenum];
		outputdata=new double[outputnodenum];
		rightoutput=new double[outputnodenum];
		map=new HashMap<String, Integer>();
		classname=new String[]{"Iris-setosa","Iris-versicolor","Iris-virginica"};
		for(int i=0;i<3;++i)
			map.put(classname[i], i);
		for(int i=0;i<inputnodenum;++i)
		{
			for(int j=0;j<hidennodenum;++j)
			{
				v[i][j]=Math.random();
			}
		}
		for(int i=0;i<hidennodenum;++i)
		{
			for(int j=0;j<outputnodenum;++j)
			{
				w[i][j]=Math.random();
			}
		}
	}
	
	void showweight()
	{
		System.out.println("V:");
		for(int i=0;i<inputnodenum;++i)
		{
			for(int j=0;j<hidennodenum;++j)
			{
				System.out.print(v[i][j]+" ");
			}
			System.out.println();
		}
		System.out.println("W:");
		for(int i=0;i<hidennodenum;++i)
		{
			for(int j=0;j<outputnodenum;++j)
			{
				System.out.print(w[i][j]+" ");
			}
			System.out.println();
		}
		System.out.println();
	}
	
	double sigmoid(double x)
	{
		return 1/(1+Math.exp(-x));
	}
		
	void train() throws FileNotFoundException
	{
		String input="";
		String inputline;
		String tmpdata[];
		Scanner jin;
		jin = new Scanner(new File("iris/bezdekIris.data"));
		
		
		//FileInputStream stream = new FileInputStream(new File("iris/bezdekIris.data"));
		BufferedReader read = new BufferedReader(new FileReader(new File("iris/bezdekIris.data")));
		try {
			read.readLine();//和ihanhhbuh
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		
		while(jin.hasNext())
		{
			inputline=jin.nextLine();
			input+=inputline+"\r\n";
		}
		for(step=1;step<=maxstep;++step)
		{
			jin = new Scanner(input);
			
			
			
			while(jin.hasNext())
			{
				inputline=jin.next();
				tmpdata=inputline.split(",");
				// 解析数据
				for(int i=0;i<inputnodenum;++i)
				{//构造输入向量 X{ x1 ， x2 ， x3 ， x4 }
					inputdata[i]=Double.parseDouble(tmpdata[i]);
				}
				//构造 期望值 Y { y1 ， y2 ， y3 ， y4 }
				for(int i=0;i<3;++i)
				{
					if(i==map.get(tmpdata[inputnodenum]))
						rightoutput[i]=1;
					else rightoutput[i]=0;
				}			
				
				//计算隐藏层  1次
				for(int j=0;j<hidennodenum;++j)//1--》3
				{
					hidendata[j]=-theta;//theta ： 是 偏移量  
					for(int i=0;i<inputnodenum;++i)//计算第j层的节点的值  1--》4
						hidendata[j]+=v[i][j]*inputdata[i];
					hidendata[j]=sigmoid(hidendata[j]);//获取节点 H（i）
				}
				
				//计算输出层  2次 
				for(int k=0;k<outputnodenum;++k)//4 
				{
					outputdata[k]=-theta;
					for(int j=0;j<hidennodenum;++j)//3
						outputdata[k]+=w[j][k]*hidendata[j];
					outputdata[k]=sigmoid(outputdata[k]);
				}
				
				//计算 权值的 偏移量 
				for(int k=0;k<outputnodenum;++k)//
				{
					for(int j=0;j<hidennodenum;++j)//
						deltaw[step][j][k]=-eta*(-(rightoutput[k]-outputdata[k]))*(1-outputdata[k])*outputdata[k]*hidendata[j];
				}
				
				// 按照公式计算出 节点值得变一辆 
				for(int j=0;j<hidennodenum;++j)
				{
					for(int i=0;i<inputnodenum;++i)
					{
						double sum=0;
						for(int k=0;k<outputnodenum;++k)
							sum+=-(rightoutput[k]-outputdata[k])*(1-outputdata[k])*outputdata[k]*w[j][k];
						deltav[step][i][j]=-eta*sum*(1-hidendata[j])*hidendata[j]*inputdata[i];
					}
				}
				// 对权重进行修改  
				for(int k=0;k<outputnodenum;++k)//
				{
					for(int j=0;j<hidennodenum;++j)
						w[j][k]+=deltaw[step][j][k]+alpha*deltaw[step-1][j][k];
				}
				//对 节点值 进行修改    alpha是学习率  ， 自定义 
				for(int j=0;j<hidennodenum;++j)
				{
					for(int i=0;i<inputnodenum;++i)
						v[i][j]+=deltav[step][i][j]+alpha*deltav[step-1][i][j];
				}
			}
		}
	}
	
	void test() throws FileNotFoundException
	{
		int total=0;
		int right=0;
		String inputline;
		String tmpdata[];
		Scanner jin = new Scanner(new File("iris/iris.data"));
		while(jin.hasNext())
		{
			inputline=jin.next();
			tmpdata=inputline.split(",");
			for(int i=0;i<inputnodenum;++i)
			{
				inputdata[i]=Double.parseDouble(tmpdata[i]);
			}
			
			for(int j=0;j<hidennodenum;++j)//
			{
				hidendata[j]=-theta;
				for(int i=0;i<inputnodenum;++i)
					hidendata[j]+=v[i][j]*inputdata[i];
				hidendata[j]=sigmoid(hidendata[j]);
			}
			
			for(int k=0;k<outputnodenum;++k)//
			{
				outputdata[k]=-theta;
				for(int j=0;j<hidennodenum;++j)
					outputdata[k]+=w[j][k]*hidendata[j];
				outputdata[k]=sigmoid(outputdata[k]);
			}
			
			int classid=0;
			
			for(int k=1;k<outputnodenum;++k)
			{
				if(outputdata[classid]<outputdata[k])
					classid=k;
			}
			System.out.print(inputline+" "+classname[classid]+" ");
			if(classid==map.get(tmpdata[inputnodenum]))
			{
				System.out.println("right");
				right++;
			}
			else System.out.println("wrong");
			total++;
		}
		System.out.println();
		System.out.println("The total number of test data:"+total);
		System.out.println("The right number of test data:"+right);
		System.out.println("The right ratio: of test data:"+(double)right/total);
	}
	
	void run() throws FileNotFoundException
	{
		long t1=System.currentTimeMillis();
		init();
		train();
		test();
		System.out.println("Runtime:"+(System.currentTimeMillis()-t1)+"ms");
	}
}
