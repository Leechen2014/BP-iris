package backp;

import java.io.*;
import java.util.Scanner;


public class backpro {
	public static void main(String args[])
	{
		String filename=new String("delta.in");
		try {			
			FileInputStream fileInputStream=new FileInputStream(filename);
			Scanner sinScanner=new Scanner(fileInputStream);
			int attN,hidN,outN,samN;
			attN=sinScanner.nextInt();
			outN=sinScanner.nextInt();
			hidN=sinScanner.nextInt();
			samN=sinScanner.nextInt();
			//System.out.println(attN+" "+outN+" "+hidN+" "+samN);
			double samin[][]=new double[samN][attN];
			double samout[][]=new double[samN][outN];
			for(int i=0;i<samN;++i)
			{
				for(int j=0;j<attN;++j)
				{
					samin[i][j]=sinScanner.nextDouble();
				}
				for(int j=0;j<outN;++j)
				{
					samout[i][j]=sinScanner.nextDouble();
				}
			}
			int times=10000;
			double rate=0.5;
			BP2 bp2=new BP2(attN,outN,hidN,samN,times,rate);
			bp2.train(samin, samout);
			for(int i=0;i<hidN;++i)
			{
				for(int j=0;j<attN;++j)
					System.out.print(bp2.dw1[i][j]+" ");
				System.out.println();
			}
			for(int i=0;i<outN;++i)
			{
				for(int j=0;j<hidN;++j)
					System.out.print(bp2.dw2[i][j]+" ");
				System.out.println();
			}
			while(true)
			{
				double testout[]=new double[outN];
				double testin[]=new double[attN];
				Scanner testinScanner=new Scanner(System.in);
				for(int i=0;i<attN;++i)
				{
					testin[i]=testinScanner.nextDouble();
				}
				testout=bp2.getResault(testin);
				for(int i=0;i<outN;++i)
					System.out.print(testout[i]+" ");
				System.out.println(outN);
			}
		} catch (IOException e) {
			// TODO: handle exception
		}
		System.out.println("End");
	}
}
class BP2//����һ���������������
{
	double dw1[][],dw2[][];
	int hidN;//�����㵥Ԫ����
	int samN;//ѧϰ��������
	int attN;//���뵥Ԫ����
	int outN;//�����Ԫ����
	int times;//��������
	double rate;//ѧϰ����
	boolean trained=false;//��֤�ڵý��ǰ����ѵ��
	BP2(int attN,int outN,int hidN,int samN,int times,double rate)
	{
		this.attN=attN;
		this.outN=outN;
		this.hidN=hidN;
		this.samN=samN;
		dw1=new double[hidN][attN+1];//ÿ�����һ������ֵw0
		for(int i=0;i<hidN;++i)//ÿ�д����������뵽i���ص�Ԫ��Ȩֵ
		{			
			for(int j=0;j<=attN;++j)
				dw1[i][j]=Math.random()/2;
		}
		dw2=new double[outN][hidN+1];//�����Ȩֵ,ÿ�����һ������ֵw0
		for(int i=0;i<outN;++i)//ÿ�д����������ص�Ԫ��i�����Ԫ��Ȩֵ
		{			
			for(int j=0;j<=hidN;++j)
				dw2[i][j]=Math.random()/2;
		}
		this.times=times;
		this.rate=rate;
	}
	public void train(double samin[][],double samout[][])
	{
		double dis=0;//�������
		int count=times;
		double temphid[]=new double[hidN];
		double tempout[]=new double[outN];
		double wcout[]=new double[outN];
		double wchid[]=new double[hidN];
		while((count--)>0)//����ѵ��
		{
			dis=0;
			for(int i=0;i<samN;++i)//����ÿ������ samin[i]
			{
				for(int j=0;j<hidN;++j)//����ÿ�������㵥Ԫ�Ľ��
				{
					temphid[j]=0;
					for(int k=0;k<attN;++k)
						temphid[j]+=dw1[j][k]*samin[i][k];
					temphid[j]+=dw1[j][attN];//������ֵ��������������
					temphid[j]=1.0/(1+Math.exp(-temphid[j] ));
				}
				for(int j=0;j<outN;++j)//����ÿ������㵥Ԫ�Ľ��
				{
					tempout[j]=0;
					for(int k=0;k<hidN;++k)
						tempout[j]+=dw2[j][k]*temphid[k];
					tempout[j]+=dw2[j][hidN];//������ֵ������������
					tempout[j]=1.0/(1+Math.exp( -tempout[j] ));
				}
				//����ÿ�������Ԫ�������
				
				for(int j=0;j<outN;++j)
				{
					wcout[j]=tempout[j]*(1-tempout[j])*(samout[i][j]-tempout[j]);
					dis+=Math.pow((samout[i][j]-tempout[j]),2);
				}
				//����ÿ�����ص�Ԫ�������
				
				for(int j=0;j<hidN;++j)
				{
					double wche=0;
					for(int k=0;k<outN;++k)//�������������
					{
						wche+=wcout[k]*dw2[k][j];
					}
					wchid[j]=temphid[j]*(1-temphid[j])*wche;
				}
				//�ı�������Ȩֵ
				for(int j=0;j<outN;++j)
				{
					for(int k=0;k<hidN;++k)
					{
						dw2[j][k]+=rate*wcout[j]*temphid[k];
					}
					dw2[j][hidN]=rate*wcout[j];
				}
				//�ı��������Ȩֵ
				for(int j=0;j<hidN;++j)
				{
					for(int k=0;k<attN;++k)
					{
						dw1[j][k]+=rate*wchid[j]*samin[i][k];
					}
					dw1[j][attN]=rate*wchid[j];
				}

			}
			if(dis<0.003)
				break;
		}
		trained=true;
	}
	
	public double[] getResault(double samin[])
	{
		double temphid[]=new double[hidN];
		double tempout[]=new double[outN];
		if(trained==false)
			return null;

		for(int j=0;j<hidN;++j)//����ÿ�������㵥Ԫ�Ľ��
		{
			temphid[j]=0;
			for(int k=0;k<attN;++k)
				temphid[j]+=dw1[j][k]*samin[k];
			temphid[j]+=dw1[j][attN];//������ֵ��������������
			temphid[j]=1.0/(1+Math.exp(-temphid[j] ));
		}
		for(int j=0;j<outN;++j)//����ÿ������㵥Ԫ�Ľ��
		{
			tempout[j]=0;
			for(int k=0;k<hidN;++k)
				tempout[j]+=dw2[j][k]*temphid[k];
			tempout[j]+=dw2[j][hidN];//������ֵ������������
			tempout[j]=1.0/(1+Math.exp( -tempout[j]));			
			//System.out.print(tempout[j]+" ");			
		}
		return tempout;		
	}
}
